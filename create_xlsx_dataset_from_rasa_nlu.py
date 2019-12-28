from rasa.nlu.training_data.formats import MarkdownReader
import xlsxwriter

workbook = xlsxwriter.Workbook('filename.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write('A1', 'question')
worksheet.write('B1', 'label')
worksheet.write('C1', 'answer')

training_data = ()
row = 1
col = 0

doc = "PATH\\TO\\nlu.md"

reader = MarkdownReader()
reader.read(doc, language='de', fformat='MARKDOWN')
for message in reader.training_examples:
    training_data = training_data + ([message.text, message.get('intent')],)

for question, label in (training_data):
    worksheet.write_string(row, col, question)
    worksheet.write_string(row, col + 1, label)
    worksheet.write_string(row, col + 2, '')
    row += 1

workbook.close()



