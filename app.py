import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import tempfile

from structure_generator_based_on_r_group_ga_multi_y_gtmr import generate_structure


# 目標範囲を設定するための関数
def input_target_ranges():
    st.subheader("目標範囲の設定")
    target_ranges = pd.DataFrame(
        columns=["y1", "y2", "y3"], index=["lower_limit", "upper_limit"]
    )

    # 目標範囲の入力フォーム
    for column in target_ranges.columns:
        cols = st.columns(len(target_ranges.columns))
        lower, upper = st.slider(
            f"範囲 for {column}",
            min_value=0.0,  # スライダーの最小値
            max_value=100.0,  # スライダーの最大値
            value=(0.0, 100.0),  # 初期値
        )
        target_ranges.at["lower_limit", column] = lower
        target_ranges.at["upper_limit", column] = upper
    return target_ranges


def convert_smiles_to_mol(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        img = Draw.MolToImage(molecule)
        st.image(img, caption="化学構造", width=384)


# 化学構造を生成する関数
def generate_structures(target_ranges):
    number_of_structures = 10  # デモのために数を減らす
    # ... ここに構造生成コードを組み込む ...
    # generated_structures = ["CCO", "CNC"]  # デモのためにダミーの構造
    # # 生成した構造をファイルに書き出す
    # str_ = "\n".join(generated_structures)
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".smi") as tmpfile:
    #     writer = open(tmpfile.name, "w")
    #     writer.write(str_)
    #     writer.close()
    #     return tmpfile.name
    generated_data = generate_structure()
    st.dataframe(generated_data[:number_of_structures])
    for smiles in generated_data[:number_of_structures].index:
        convert_smiles_to_mol(smiles)
    return generated_data[:number_of_structures]


# Streamlitアプリのメイン関数
def main():
    st.title("化学構造生成ツール")

    # 目標範囲を入力
    target_ranges = input_target_ranges()

    # 生成ボタン
    if st.button("構造生成"):
        file_path = generate_structures(target_ranges)
        # st.success(f"生成された構造は {file_path} に保存されました。")
        # st.download_button("ダウンロード", file_path, file_name="generated_structures.smi")


if __name__ == "__main__":
    main()
