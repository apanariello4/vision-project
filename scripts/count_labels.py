import glob

text_files_path = "images/*.txt"
file_list = glob.glob(text_files_path)

paintings_number, statues_number = 0, 0
for label_file in file_list:
    with open(label_file, 'r') as f:
        lines = [line.rstrip() for line in f]
        for line in lines:
            if line[0] == '0':
                paintings_number += 1
            elif line[0] == '1':
                statues_number += 1

print(f'Labeled Images: {len(file_list)}\nTotal Labels: {paintings_number + statues_number}\nPainting Labels: {paintings_number}\n Statue Labels: {statues_number}')
