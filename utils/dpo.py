import torch
import torch.nn.functional as F

def compute_dpo_loss(
    model_chosen_logprobs,
    model_rejected_logprobs,
    reference_chosen_logprobs,
    reference_rejected_logprobs,
    beta=0.1,
):
    """
    计算DPO损失。
    参数：
        model_chosen_logprobs：模型选择的对数概率。shape：(batch_size,)
        model_rejected_logprobs：模型拒绝的对数概率。shape：(batch_size,)
        reference_chosen_logprobs：参考模型选择的对数概率。shape：(batch_size,)
        reference_rejected_logprobs：参考模型拒绝的对数概率。shape：(batch_size,)
        beta：DPO损失的温度参数。默认值：0.1

    返回：
        损失、选择奖励、拒绝奖励
    """
    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    # DPO损失
    losses = -F.logsigmoid(beta * logits)

    # 可选值，用于跟踪训练进度
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    # 计算平均值
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

def compute_logprobs(logits, labels, selection_mask=None):
    """
    计算对数概率。

    参数：
        logits：shape为(batch_size, num_tokens, vocab_size)
        labels：shape为(batch_size, num_tokens)
        selection_mask：shape为(batch_size, num_tokens)

    返回：
        平均对数概率（不包括填充padding）
    """

    # 标签是输入右移一位
    labels = labels[:, 1:].clone()

    # 截断logits以匹配标签的num_tokens
    logits = logits[:, :-1, :]

    log_probs = F.log_softmax(logits, dim=-1)

    # 收集实际标签的对数概率
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone()

        # 应用掩码以过滤填充padding
        selected_log_probs = selected_log_probs * mask

        # 计算平均对数概率（不包括填充padding）
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)

        return avg_log_prob

    else:
        return selected_log_probs.mean(-1)

def compute_dpo_loss_batch(batch, policy_model, reference_model, beta):
    """
    计算批次的DPO损失。

    参数：
        batch：批次数据
        policy_model：策略模型
        reference_model：参考模型
        beta：DPO损失的温度参数

    返回：
        损失、选择奖励、拒绝奖励
    """

    policy_chosen_logprobs = compute_logprobs(
        logits=policy_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"]
    )
    policy_rejected_logprobs = compute_logprobs(
        logits=policy_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"]
    )
    reference_chosen_logprobs = compute_logprobs(
        logits=reference_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"]
    )
    reference_rejected_logprobs = compute_logprobs(
        logits=reference_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"]
    )
    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_logprobs,
        model_rejected_logprobs=policy_rejected_logprobs,
        reference_chosen_logprobs=reference_chosen_logprobs,
        reference_rejected_logprobs=reference_rejected_logprobs,
        beta=beta
    )
    return loss, chosen_rewards, rejected_rewards

def compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta, num_batches=None):
    """
    将计算DPO损失批次应用于整个数据加载器。

    参数：
        data_loader：数据加载器
        policy_model：策略模型
        reference_model：参考模型
        beta：DPO损失的温度参数
        num_batches：批次数

    返回：
        总损失、总选择奖励、总拒绝奖励
    """

    total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
    if len(data_loader) == 0:
        return float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果批次数超过数据加载器中的批次数，则减少批次数
        num_batches = min(num_batches, len(data_loader))
    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()

        else:
            break

    # 计算平均值
    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    return total_loss, total_chosen_rewards, total_rejected_rewards

def evaluate_dpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):
    """
    计算训练和验证数据集的DPO损失。
    参数：
        policy_model：策略模型
        reference_model：参考模型
        train_loader：训练数据加载器
        val_loader：验证数据加载器
        beta：DPO损失的温度参数
        eval_iter：评估迭代次数

    返回：
        训练损失、训练选择奖励、训练拒绝奖励、验证损失、验证选择奖励、验证拒绝奖励
    """
    policy_model.eval()
    with torch.no_grad():
        train_loss, train_chosen_rewards, train_rejected_rewards = compute_dpo_loss_loader(
            data_loader=train_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

        val_loss, val_chosen_rewards, val_rejected_rewards = compute_dpo_loss_loader(
            data_loader=val_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

    res = {
        "train_loss": train_loss,
        "train_chosen_reward": train_chosen_rewards,
        "train_rejected_reward": train_rejected_rewards,
        "val_loss": val_loss,
        "val_chosen_reward": val_chosen_rewards,
        "val_rejected_reward": val_rejected_rewards
    }

    policy_model.train()
    return res