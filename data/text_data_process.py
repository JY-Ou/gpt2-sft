from pathlib import Path

import pandas as pd


def create_balanced_dataset(df):
    # 统计"spam"的实例数量
    num_spam = df[df["Label"] == "spam"].shape[0]

    # 随机采样"ham"实例,数量与"spam"实例相同
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # 将"ham"子集与"spam"合并
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df

def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df



if __name__ == '__main__':
    data_file_path = Path("sms_spam_collection") / "SMSSpamCollection.tsv"
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    print(balanced_df["Label"].value_counts())

    # 将平衡后的数据集保存为CSV文件
    balanced_df.to_csv(Path("sms_spam_collection") /'balanced_dataset.csv', index=False)
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    train_df.to_csv(Path("sms_spam_collection") /"train.csv", index=None)
    validation_df.to_csv(Path("sms_spam_collection") /"val.csv", index=None)
    test_df.to_csv(Path("sms_spam_collection") /"test.csv", index=None)