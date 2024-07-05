# # 這是讀取trigger set的

import sys
import re
import pandas as pd

def extract_acc_from_output(file_path):
    nat_acc_values = []
    query_acc_values = []

    with open(file_path, 'r') as file:
        for line in file:
            nat_match = re.search(r"Best valid nat acc: (\d+\.\d+)", line)
            if nat_match:
                nat_acc_values.append(float(nat_match.group(1)))
            query_match = re.search(r"Best valid query acc: (\d+\.\d+)", line)
            if query_match:
                query_acc_values.append(float(query_match.group(1)))

    return nat_acc_values, query_acc_values

def write_to_excel(output_file):
    # 提取 nat acc 和 query acc
    nat_acc_values, query_acc_values = extract_acc_from_output(output_file)

    # 创建一个 DataFrame
    df = pd.DataFrame({
        'Nat Acc': nat_acc_values,
        'Query Acc': query_acc_values
    })

    # 将 DataFrame 写入 Excel 文件
    excel_file = 'outputs.xlsx'
    df.to_excel(excel_file, index=False)

    print(f"Outputs have been written to {excel_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python write_to_excel.py <output_file>")
        sys.exit(1)

    output_file = sys.argv[1]
    write_to_excel(output_file)


# 這是讀取my way的
# import sys
# import re
# import pandas as pd

# def extract_metrics_from_output(file_path):
#     mse_values = []
#     val_acc_values = []

#     with open(file_path, 'r') as file:
#         for line in file:
#             mse_match = re.search(r"Test MSE loss: (\d+\.\d+)", line)
#             if mse_match:
#                 mse_values.append(float(mse_match.group(1)))
#             val_acc_match = re.search(r"Val Acc: (\d+\.\d+)", line)
#             if val_acc_match:
#                 val_acc_values.append(float(val_acc_match.group(1)))

#     return mse_values, val_acc_values

# def write_to_excel(output_file):
#     # 提取 Test MSE loss 和 Val Acc
#     mse_values, val_acc_values = extract_metrics_from_output(output_file)

#     # 创建一个 DataFrame
#     df = pd.DataFrame({
#         'Test MSE Loss': mse_values,
#         'Val Acc': val_acc_values
#     })

#     # 将 DataFrame 写入 Excel 文件
#     excel_file = 'outputs.xlsx'
#     df.to_excel(excel_file, index=False)

#     print(f"Outputs have been written to {excel_file}")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python write_to_excel.py <output_file>")
#         sys.exit(1)

#     output_file = sys.argv[1]
#     write_to_excel(output_file)



