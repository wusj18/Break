import os
import re
import csv
import json


class data_format_trans(object):
    def __init__(self):
        pass

    def special_token(self, sen):
        replace_dic = {"#10": "ú", "#11": "û", "#12": "ü", "#13": "ý", "#14": "þ", "#15": "ÿ", "#16": "Ā", "#17": "ā", "#18": "Ă", "#19": "ă", "#1": "À", "#2": "Á", "#3": "ñ", "#4": "ò", "#5": "õ", "#6": "ö", "#7": "÷", "#8": "ø", "#9": "ù"}
        for k, v in replace_dic.items():
            sen = sen.replace(v, k)
        return sen

    def replace_nums(self, s):
        s = re.sub(r'#(\d)', r'@@\1@@', s)
        # s = s.replace("return ", "")
        return s

    def csv2srctgt(self, data_dir, csvfile):
        prefix = csvfile.split(".")[0]
        max_num = 0

        with open(data_dir + csvfile, 'r') as f, open(data_dir + prefix + ".source", 'w') as fsource, open(data_dir + prefix + ".target", "w") as ftarget:
            reader = csv.reader(f)
            sub_ques_num_dic = {}
            lines_num = 0
            for row in reader:
                if lines_num == 0:
                    lines_num += 1
                    continue
                if len(row) == 1:
                    src = row[0]
                    tgt = row[0]
                # if "," in row[2]:
                #     print(row[2])
                else:
                    src = self.special_token(row[1])
                    tgt = re.sub(r'@@(\d?)@@', r'#\1', row[2])
                    tgt = self.special_token(tgt) #.split(";")
                
                fsource.write(src + "\n")
                if len(row) > 3:
                    opt = row[3][1:-1].replace("'", "").split(", ")
                
                    symb_re = re.compile(r'#\d+')
                    sub_questions = tgt.split(";")
                    if len(sub_questions) > 2:
                        pass
                    for index, sub_q in enumerate(sub_questions):
                        sym = re.findall(symb_re, sub_q)
                        if sym:
                            opt[index] += " " + " ".join(sym)

                # tgt = ";".join(sub_ques)
                # max_num = max(len(sub_ques), max_num)

                    ftarget.write(" ".join(opt) + " <\s> " + tgt + "\n")
                else:
                    ftarget.write(tgt + "\n")
                lines_num += 1

                sub_ques = len(tgt.split(";"))
                if sub_ques in sub_ques_num_dic.keys():
                    sub_ques_num_dic[sub_ques] += 1
                else:
                    sub_ques_num_dic[sub_ques] = 1

                # if max_num > 10:
                #     print(tgt)
                #     max_num = 0
            print(prefix + str(lines_num))
            print(max_num)
            print(sorted(sub_ques_num_dic.items(), key=lambda d: d[1]))

    def case_study(self, data_dir, csvfile):
        prefix = csvfile.split(".")[0]
        miss_words_dic = {}
        with open(data_dir + csvfile, 'r') as f:
            reader = csv.reader(f)
            # print(type(reader))
            lines_num = 0
            for row in reader:
                if lines_num == 0:
                    lines_num += 1
                    continue
                pred_words = self.special_token(row[1]).split(" ")
                target_words = self.special_token(row[2]).split(" ")
                # miss_words = [x for x in target_words if x not in pred_words]
                miss_words = [x for x in pred_words if x not in target_words]

                miss_words = list(set(miss_words))
                for w in miss_words:
                    if w not in miss_words_dic:
                        miss_words_dic[w] = 1
                    else:
                        miss_words_dic[w] += 1
            print(sorted(miss_words_dic.items(), key=lambda d: d[1]))
                
    def lexicon_words(self, file_name, tgt_file):
        seen = {}
        print(file_name)
        with open(file_name, "r", encoding="utf-8") as f:
                # , open(tgt_file) as tgtf:
            # lines = json.load(f)
            # for line, tgt in zip(f, tgtf):
            for line in f:
                line = json.loads(line)
                special = []
                words = line["allowed_tokens"][1:-1].replace(" '", "").replace("'", "").split(",")
                # words = line["allowed_tokens"]
                # ques = line["source"]
                # tgt = tgt.strip().split(" ")
                for w in words:
                    if w in seen:
                        seen[w] += 1
                    else:
                        seen[w] = 1
            words_stat = sorted(seen.items(), key=lambda x: x[1], reverse=True)
            words_stat = [x[0] for x in words_stat if x[1] >= 3100]
            words_freq = sorted(words_stat)
        with open(file_name, "r", encoding="utf-8") as f, open(file_name.rstrip(".json") + "_special.json", "w", encoding="utf-8") as fout:
            words_max_num = 0
            for line in f:
                line = json.loads(line)
                words = line["allowed_tokens"][1:-1].replace(" '", "").replace("'", "").split(",")
                sorted_words = []
                freq_words = set()
                spec_words = set()
                for w in words:
                    if w in words_freq:
                        freq_words.add(w)
                    else:
                        spec_words.add(w)
                # sorted_words = sorted(freq_words) + sorted(spec_words)
                sorted_words = sorted(spec_words)
                line["allowed_tokens"] = sorted_words
                words_max_num = max(len(sorted_words), words_max_num)
                fout.write(json.dumps(line) + "\n")
                    # if w not in words:
                    #     print(w)
                # print("-----------")
                for w in words:
                    if w not in seen:
                        seen.add(w)
                        special.append(w)
                    else:
                        continue
                print(line["source"], special)
            print(words_max_num)

    def spacy_csv(self, data_dir, csvfile):
        prefix = csvfile.split(".")[0]
        with open(data_dir + csvfile, 'r') as f:
            reader = csv.reader(f)
            lines_num = 0
            for row in reader:
                if lines_num == 0:
                    lines_num += 1
                    continue
                if len(row) < 1:
                    continue
                question = row[1]
                decomp = row[2]
                import spacy
                nlp = spacy.load('en')
                question = nlp(question)
                decomp = nlp(decomp)
                for token in question:
                    print(token, token.pos_, token.pos)
                print("-" * 20)
                for token in decomp:
                    print(token, token.pos_, token.pos)
                print("*" * 20)
                for token in question.ents:
                    print(token, token.label_, token.label)
                print("-" * 20)
                for token in decomp.ents:
                    print(token, token.label_, token.label)
                print("*" * 20)
                for token in question.noun_chunks:
                    print(token)
                print("-" * 20)
                for token in decomp.noun_chunks:
                    print(token)
                print("*" * 20)
                    
                break
                     

def data_trans(data_dir):
    dft = data_format_trans()
    for file in os.listdir(data_dir):
        if file.endswith("csv"):
            dft.csv2srctgt(data_dir, file)
            # if "train.csv" == file:
            # dft.case_study(data_dir, file)
            #     dft.spacy_csv(data_dir, file)
        if file.endswith("json") and "sorted" not in file:
            dft.lexicon_words(data_dir + file, data_dir + file.split("_")[0] + ".target")


if __name__ == '__main__':
    data_trans("../QDMR_high_opts_symbol/")
