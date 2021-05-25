import pandas as pd

data = pd.read_csv('results.csv')
print(data)
# rouge_l = data.loc[data['ROUGE-Type']=='ROUGE-L']
# rouge_1 = data.loc[data['ROUGE-Type']=='ROUGE-1']
# rouge_2 = data.loc[data['ROUGE-Type']=='ROUGE-2']
# rouge_s = data.loc[data['ROUGE-Type']=='ROUGE-SU4']

rouge_l = data.loc[data['ROUGE-Type'].str.contains('ROUGE-L')]
rouge_1 = data.loc[data['ROUGE-Type'].str.contains('ROUGE-1')]
rouge_2 = data.loc[data['ROUGE-Type'].str.contains('ROUGE-2')]
rouge_s = data.loc[data['ROUGE-Type'].str.contains('ROUGE-SU4')]

with pd.ExcelWriter('output.xlsx') as writer:
    rouge_l.to_excel(writer, sheet_name='ROUGE-L')
    rouge_1.to_excel(writer, sheet_name='ROUGE-1')
    rouge_2.to_excel(writer, sheet_name='ROUGE-2')
    rouge_s.to_excel(writer, sheet_name='ROUGE-S')