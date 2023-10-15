import json
from pyvi import ViTokenizer
import nltk
from nltk.tokenize import word_tokenize
from underthesea import word_tokenize as segmentation
import re
import unicodedata

# nltk.download('punkt')
files = ['./data_2/train.txt', './data_2/dev.txt', './data_2/test.txt']
file_path = "../data/Smartphone-COQE/train.txt"


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    paragraphs = text.strip().split('\n\n')

    sent_col = []
    sent_label_col = []
    final_label_col = []
    for paragraph in paragraphs:
        lines = paragraph.strip().split('\n')
        # sentence = lines[0].strip()
        sentence = lines[0].strip().split('\t')[1]
        label_col = []
        if(len(lines) == 1):
            label = 0
            dict_empty = {"subject": [], "object": [], "aspect": [], "predicate": [], "label": []}
            label_col.append(dict_empty)
        else:
            label = 1
            for i in range(len(lines)):
                if(i != 0):
                    dictionary = json.loads(lines[i])
                    label_col.append(dictionary)

        sent_col.append(sentence)
        sent_label_col.append(label)
        final_label_col.append(label_col)

    return sent_col, sent_label_col, final_label_col

def extract_indices(label_quin):
    global_elem_col = {}
    each_tuple_pair = []
    for key, values in label_quin.items():
        if key != 'preference':
            global_elem_col[key] = set()
            index_list = [value.split('&&')[0] for value in values if value]  # index for each element
            if len(label_quin['predicate']):  # check if sentence is a com sentence
                if key != 'predicate':  # 3 first elem
                    if len(index_list):
                        global_elem_col[key].add((int(index_list[0]) - 1, int(index_list[-1])))
                        each_tuple_pair.append((int(index_list[0]) - 1, int(index_list[-1])))
                    else:
                        global_elem_col[key] = ()
                        each_tuple_pair.append((-1, -1))
                else:
                    cp = 2  # COM SUP DIF
                    if label_quin['preference'][0] in ('COM+', 'SUP+'):
                        cp = 1
                    elif label_quin['preference'][0] == 'EQL':
                        cp = 0
                    elif label_quin['preference'][0] in ('COM-', 'SUP-'):
                        cp = -1
                    global_elem_col[key].add((int(index_list[0]) - 1, int(index_list[-1]), cp))
                    each_tuple_pair.append((int(index_list[0]) - 1, int(index_list[-1])))
                    each_tuple_pair.append((cp, cp))
            else:
                each_tuple_pair = [(-1, -1)] * 5
    return global_elem_col, each_tuple_pair


def idx_presentation(label_col):
    final_label_col = []
    final_tuple_pair = []
    for label_sentence in label_col:
        tuple_pair = []
        label_sentence_data = {'subject': set(), 'object': set(), 'aspect': set(), 'predicate': set()}
        for label_quin in label_sentence:
            extracted_item, each_tuple_pair = extract_indices(label_quin)
            tuple_pair.append(each_tuple_pair)
            for key in extracted_item:
                label_sentence_data[key].update(extracted_item[key])
        final_label_col.append(label_sentence_data)
        final_tuple_pair.append(tuple_pair)
    return final_label_col, final_tuple_pair


sent_col, sent_label_col, label_col = read_file(file_path=file_path)

label_col, tuple_pair_col = idx_presentation(label_col)

# for i in range(len(label_col)):
#     if sent_col[i] == "Như đã nói từ đầu thì phong cách thiết kế của Samsung Galaxy Note 10 Lite có nhiều điểm mới lạ so với các dòng máy tiền nhiệm. ":


for i in range(len(sent_col)):
    sent_col[i] = re.sub(r'\.\.\.', ' . . . ', sent_col[i])
    sent_col[i] = re.sub(r'…', '. . .', sent_col[i])  # sometimes it's … instead of ...
    tokens = word_tokenize(sent_col[i])
    new_tokens = []
    for token in tokens:
        # if re.match(r'\w+/\d+\.\d+', token) or re.match(r'\d+/\d+', token) or re.match(r'\d+/', token):
        #     fractions = re.split(r'(/)', token)
        #     new_tokens.extend(fractions)

        # cover all cases like above
        if '/' in token:
            fractions = re.split(r'(/)', token)
            new_tokens.extend(fractions)
        elif re.match(r"'\w+", token):  # match 'word
            parts = re.split(r"(')", token)
            new_tokens.extend(parts)
        elif re.match(r'\d+,\d+:\d+', token) or re.match(r'\d+:\d+', token):
            parts = re.split(r'(:)', token)
            new_tokens.extend(parts)
        elif re.match(r'\d+-\w+', token):
            parts = re.split(r'(-)', token)
            new_tokens.extend(parts)
        elif re.match(r'\d+\+', token):  # Check if the token matches the pattern "digit+"
            parts = re.split(r'(\+)', token)  # Split the token using "+"
            new_tokens.extend(parts)
        elif re.match(r'\w+\+', token):
            match = re.match(r'(\w+)\+', token)
            word = match.group(1)
            new_tokens.extend([word, '+'])
        elif token == '``' or token == "''":
            token = '"'
            new_tokens.append(token)
        else:
            new_tokens.append(token)
    # Chuyển về unicode tổng hợp
    new_tokens = [unicodedata.normalize("NFKC", token) for token in new_tokens if token != '']
    new_sent_col.append(new_tokens)

# print(*new_sent_col, sep='\n')

segmented_sent_col = []

# trường hợp từ bị word_segmentation bị giao với các ele thì tách nó ra ko cho cùng 1 token nữa
for i in range(len(sent_col)):
    tokens = ViTokenizer.tokenize(sent_col[i]).split()
    subject, object, aspect, predicate = [], [], [], []
    if any(len(value) > 0 for value in label_col[i].values()):
        for key in label_col[i].keys():
            values = label_col[i][key]
            for value in values:
                s_idx = value[0]
                e_idx = value[1]
                cur_e = '_'.join(new_sent_col[i][s_idx:e_idx])
                if key == 'subject':
                    subject.append(cur_e)
                elif key == 'object':
                    object.append(cur_e)
                elif key == 'aspect':
                    aspect.append(cur_e)
                else:
                    predicate.append(cur_e)
    items = [subject, object, aspect, predicate]
    for item in items:
        for smaller_item in item:
            item_token = re.split(r'_', smaller_item)
            for j in range(len(tokens)):
                cur_token = re.split(r'_', tokens[j])
                if any(token in item_token for token in cur_token):
                    if all(token in item_token for token in cur_token) is False:
                        tokens[j:j + 1] = cur_token
    token_str = ' '.join(tokens)
    #for file train.txt
    new_str = token_str.replace('Type - C', 'Type-C')
    new_str = new_str.replace('S - Pen', 'S-Pen')
    new_str = new_str.replace('Li - po', 'Li-po')
    new_str = new_str.replace('micro - USB', 'micro-USB')
    new_str = new_str.replace('full - frame', 'full-frame')
    new_str = new_str.replace('on - screen', 'on-screen')
    new_str = new_str.replace('4,5 W', '4,5W')
    new_str = new_str.replace('5.000 mAh', '5.000mAh')

    # for file dev.txt
    new_str = new_str.replace('mô - đun', 'mô-đun')
    new_str = new_str.replace('Li - on', 'Li-on')

    # for file test.txt
    new_str = new_str.replace('1,5 mm', '1,5mm')

    # add this pattern since it cause error
    if re.search(r'[A-Z]\.', new_str):
        new_str = new_str[:-1]

    # Chuyển về unicode tổng hợp
    new_str = unicodedata.normalize("NFKC", new_str)
    new_list = new_str.split()
    segmented_sent_col.append(new_list)

# print(*segmented_sent_col, sep = "\n")
mapping_dict = []

# mapping idx sang câu đã word segmentation
elem_col = ['subject', 'object', 'aspect', 'label']
new_label_col = []
count = 0
for i in range(len(label_col)):
    new_quin = {'subject': set(), 'object': set(), 'aspect': set(), 'predicate': set()}
    cur_dict = {}
    if any(len(value) > 0 for value in label_col[i].values()):
        for key in label_col[i].keys():
            values = label_col[i][key]
            for value in values:
                s_idx = value[0]
                e_idx = value[1]
                cur_e = "_".join(new_sent_col[i][s_idx:e_idx])
                # print(cur_e)
                new_start_index = None
                new_end_index = None
                cur_e_tokens = cur_e.split('_')
                for j in range(len(segmented_sent_col[i])):
                    cur_len = 1
                    while j + cur_len < len(segmented_sent_col[i]) and len('_'.join(segmented_sent_col[i][j:j + cur_len]).split('_')) < len(cur_e_tokens):
                        cur_len += 1
                    if '_'.join(segmented_sent_col[i][j:j + cur_len]) == cur_e:
                        # print('_'.join(segmented_sent_col[i][j:j + cur_len]))
                        new_start_index = j
                        new_end_index = j + cur_len
                        break
                    # TODO : kiểm tra lại xem mình tokenize đứng chưa
                if new_start_index is None and new_end_index is None:
                    count += 1
                    print("-"*20)
                    print(count)
                    print(key)
                    print(cur_e)
                    print(new_sent_col[i])
                    print(label_col[i])
                    print(segmented_sent_col[i])
                    print("-" * 20)
                cur_dict[s_idx] = new_start_index
                cur_dict[e_idx] = new_end_index
                new_value = {(new_start_index, new_end_index)}
                if key == 'predicate':
                    new_value = {(new_start_index, new_end_index, value[2])}
                new_quin[key].update(new_value)
    mapping_dict.append(cur_dict)
    new_label_col.append(new_quin)
# print(*mapping_dict, sep = "\n")
# print(*new_label_col, sep = "\n")

# TODO : dùng mapping col để mapping từ tuple_idx cũ sang mới
new_tuple_pair_col = []
for i in range(len(tuple_pair_col)):
    if tuple_pair_col == [[(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1)]]:
        new_tuple_pair_col.append([[(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1)]])
    else:
        mapped_sublist_2d = []
        for tuple_pair in tuple_pair_col[i]:
            mapped_sublist = []
            for pair in tuple_pair[:4]:
                mapped_pair = []
                for value in pair:
                    if value == -1:
                        mapped_value = -1
                    else:
                        mapped_value = mapping_dict[i][value]
                    mapped_pair.append(mapped_value)
                mapped_sublist.append(tuple(mapped_pair))
            mapped_sublist.append(tuple_pair[4])
            mapped_sublist_2d.append(mapped_sublist)
        new_tuple_pair_col.append(mapped_sublist_2d)


