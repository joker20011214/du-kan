import openpyxl
wb = openpyxl.load_workbook(r"c:\Users\36347\Desktop\分指标CSV\牛津分级指标.xlsx", data_only=True)
for ws in wb.worksheets:
    print("=== SHEET:", ws.title, "dims:", ws.dimensions, "===")
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 50), values_only=True):
        print(row)
