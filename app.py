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
    #    target_ranges で配列準備
    target_ranges = []

    # 目標範囲の入力フォーム
    default_values = [1.0, -60.0, 30.0]  # デフォルト値のリスト

    value1 = st.slider(
        f"水溶解度",
        min_value=-100.0,  # スライダーの最小値
        max_value=100.0,  # スライダーの最大値
        value=default_values[0],  # 初期値
    )
    value2 = st.slider(
        f"仮想物性値1",
        min_value=-100.0,  # スライダーの最小値
        max_value=100.0,  # スライダーの最大値
        value=default_values[1],  # 初期値
    )
    value3 = st.slider(
        f"仮想物性値2",
        min_value=-100.0,  # スライダーの最小値
        max_value=100.0,  # スライダーの最大値
        value=default_values[2],  # 初期値
    )
    target_ranges.append(value1)
    target_ranges.append(value2)
    target_ranges.append(value3)
    return target_ranges


def convert_smiles_to_mol(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        img = Draw.MolToImage(molecule)
        st.image(img, caption="化学構造", width=384)


# 化学構造を生成する関数
def generate_structures(target_ranges):
    number_of_structures = 10  # デモのために数を減らす
    data = {
        "smiles": [
            "C=CC1CN2CCC1CC2C1C(NCC(OC)OC)C1(C)C",
            "C=CC1CN2CCC1CC2C1C(c2ccc3c(c2)OCO3)C1(C)C",
            "C=CC1CN2CCC1CC2C1C(C2(CC)C(=O)NC(=S)NC2=O)C1(C)C",
            "CC(C)CCOC1C(C(CCCl)c2ccccc2)C1(C)C",
            "CC1(C)C(CNC(=O)c2ccccc2)C1C1CCCCC1=O",
            "C=CC1CN2CCC1CC2C1C(Nc2ccncc2S(O)(O)NC(=O)NC(C)C)C1(C)C",
            "CCCC(C)Nc1ccccc1C1(CBr)OCCO1",
            "CCC=C1CCC2C3CCC4CC(c5ccccc5NC(C#N)c5nc(C)c(N=Nc6c(Cl)cc(Cl)cc6Cl)s5)CCC4(C)C3CCC12C",
            "CC(NC(CCc1ccccc1)C(=O)O)C(=O)c1ccccc1NCC1=C(C=NO)CCC1",
            "CC(=O)Nc1ccccc1CCNc1ccccc1NCc1ccccc1C",
        ],
        "value": [
            0.0807,
            0.0792,
            0.0561,
            0.0517,
            0.052,
            0.0524,
            0.0552,
            0.0192,
            0.0315,
            0.0421,
        ],
    }
    print(target_ranges)
    df = pd.DataFrame(data)
    # generated_data = df
    generated_data = generate_structure(target_ranges)
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
