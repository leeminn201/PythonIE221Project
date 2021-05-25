# pip install transformers
# !pip3 install sentencepiece
# !pip3 install tf_sentencepiece

from Library import Library_Structure as l
# from LoadData import load_data as load
from Load_Data import Processing_data as p
import pandas as pd
# from LoadData import processing_data as p
# from Trainning.train_model import Train
# from VisualData.visual_data import Visual
# from BuildModel import build as b
# from LoadData.load_data import MAX_LEN, PATH
# from Trainning import Test_cau as Test


from Load_Data.Processing_data import PATH, MAX_LEN
from Build_Model import build as b
from Tranning import Train
from Tranning.Train import Train
from Tranning import Test_Sentences as Test
from VisualData.visual_data import Visual


class Menu():
    def __init__(self):
        self.input_ids = p.a.input_ids
        self.input_ids_t = p.a.input_ids_t
        self.preds_start_train = l.np.zeros((self.input_ids.shape[0], 96))
        self.preds_end_train = l.np.zeros((self.input_ids.shape[0], 96))
        self.preds_start = l.np.zeros((self.input_ids_t.shape[0], 96))
        self.preds_end = l.np.zeros((self.input_ids_t.shape[0], 96))
        self.start_tokens = p.a.start_tokens
        self.end_tokens = p.a.end_tokens

    def menu1(self):
        # return load.train.shape, load.test.shape
        return p.train.shape, p.test.shape

    def menu2(self):
        # return load.train.head(30)
        return p.train.head(30)

    def menu3(self):
        # return load.test.head(30)
        return p.test.head(30)

    def menu4(self):
        x = b.Model_RoBERTa(96, PATH)
        return x.build_model()

    def menu5(self):
        Build = Train(MAX_LEN, PATH, p.a.input_ids, p.a.input_ids_t, p.a.attention_mask,
                      p.a.attention_mask_t, p.a.token_type_ids, p.a.token_type_ids_t, p.a.start_tokens, p.a.end_tokens)
        # self.input_ids, self.input_ids_t, self.preds_start_train, self.preds_end_train, self.preds_start, self.preds_end, self.start_tokens, self.end_token =
        Build.Train_model()
        return Build.Acu()

    def menu6(self):
        link = '/content/drive/MyDrive/Do_An_Python/Dataset/train.csv'
        df_train = pd.read_csv(link).fillna('')
        all = Test.a.Test_Model(df_train)
        return Test.a.Result_CSV(df_train, all, '/content/drive/MyDrive/sample_train.csv')

    def menu7(self):
        link = '/content/drive/MyDrive/Do_An_Python/Dataset/test.csv'
        df_test = pd.read_csv(link).fillna('')
        all = Test.a.Test_Model(df_test)
        return Test.a.Result_CSV(df_test, all, '/content/drive/MyDrive/sample_test.csv')

    def menu8(self):
        Visual.wordCloud()

    def menu12(self, cau):
        for c in cau:
            c = c.upper()
            if (c == "A"):
                print("..######..\n..#....#..\n..######..\n..#....#..\n..#....#..\n\n")
            elif (c == "B"):
                print("..######..\n..#....#..\n..#####...\n..#....#..\n..######..\n\n")
            elif (c == "C"):
                print("..######..\n..#.......\n..#.......\n..#.......\n..######..\n\n")
            elif (c == "D"):
                print("..#####...\n..#....#..\n..#....#..\n..#....#..\n..#####...\n\n")
            elif (c == "E"):
                print("..######..\n..#.......\n..#####...\n..#.......\n..######..\n\n")
            elif (c == "F"):
                print("..######..\n..#.......\n..#####...\n..#.......\n..#.......\n\n")
            elif (c == "G"):
                print("..######..\n..#.......\n..#####...\n..#....#..\n..#####...\n\n")
            elif (c == "H"):
                print("..#....#..\n..#....#..\n..######..\n..#....#..\n..#....#..\n\n")
            elif (c == "I"):
                print("..######..\n....##....\n....##....\n....##....\n..######..\n\n")
            elif (c == "J"):
                print("..######..\n....##....\n....##....\n..#.##....\n..####....\n\n")
            elif (c == "K"):
                print("..#...#...\n..#..#....\n..##......\n..#..#....\n..#...#...\n\n")
            elif (c == "L"):
                print("..#.......\n..#.......\n..#.......\n..#.......\n..######..\n\n")
            elif (c == "M"):
                print("..#....#..\n..##..##..\n..#.##.#..\n..#....#..\n..#....#..\n\n")
            elif (c == "N"):
                print("..#....#..\n..##...#..\n..#.#..#..\n..#..#.#..\n..#...##..\n\n")
            elif (c == "O"):
                print("..######..\n..#....#..\n..#....#..\n..#....#..\n..######..\n\n")
            elif (c == "P"):
                print("..######..\n..#....#..\n..######..\n..#.......\n..#.......\n\n")
            elif (c == "Q"):
                print("..######..\n..#....#..\n..#.#..#..\n..#..#.#..\n..######..\n\n")
            elif (c == "R"):
                print("..######..\n..#....#..\n..#.##...\n..#...#...\n..#....#..\n\n")
            elif (c == "S"):
                print("..######..\n..#.......\n..######..\n.......#..\n..######..\n\n")
            elif (c == "T"):
                print("..######..\n....##....\n....##....\n....##....\n....##....\n\n")
            elif (c == "U"):
                print("..#....#..\n..#....#..\n..#....#..\n..#....#..\n..######..\n\n")
            elif (c == "V"):
                print("..#....#..\n..#....#..\n..#....#..\n...#..#...\n....##....\n\n")
            elif (c == "W"):
                print("..#....#..\n..#....#..\n..#.##.#..\n..##..##..\n..#....#..\n\n")
            elif (c == "X"):
                print("..#....#..\n...#..#...\n....##....\n...#..#...\n..#....#..\n\n")
            elif (c == "Y"):
                print("..#....#..\n...#..#...\n....##....\n....##....\n....##....\n\n")
            elif (c == "Z"):
                print("..######..\n......#...\n.....#....\n....#.....\n..######..\n\n")
            elif (c == " "):
                print("..........\n..........\n..........\n..........\n\n")
            elif (c == "."):
                print("----..----\n\n")
            else:
                break
        return


a = Menu()
while True:
    print('===============================MENU===============================')
    print('------------< Mời bạn chọn tính năng với số tương ứng >-----------')
    print('||                                                               ||')
    print('||   1.  Xem kích thước tập train, tập test                      ||')
    print('||   2.  Xem bộ dữ liệu train                                    ||')
    print('||   3.  Xem bộ dữ liệu test                                     ||')
    print('||   4.  Xây dựng mô hình                                        ||')
    print('||   5.  Train mô hình                                           ||')
    print('||   6.  Kết quả dự đoán trên tập train                          ||')
    print('||   7.  Kết quả dự đoán trên tập test                           ||')
    print('||   8.  WordCloud                                               ||')
    print('||   9. Chạy thử trên 1 câu tự nhập                              ||')
    print('||   10. Chạy thử trên một bộ DataFrame                          ||')
    print('||   11. Lưu kết quả dưới dạng file csv                          ||')
    print('||   12. Kết thúc chương trình                                   ||')
    print('||                                                               ||')
    print('---------< Nhập số khác bất kỳ để kết thúc chương trình >----------')
    print('============================THE END===============================')
    try:
        select = int(input())
    except:
        print('Yêu cầu nhập số')
    else:
        if select == 1:
            x, y = a.menu1()
            print('Kích thước tập train: ', x)
            print('Kích thước tập test: ', y)
        elif select == 2:
            df_train = a.menu2()
            print(df_train)
        elif select == 3:
            df_test = a.menu3()
            print(df_test)
        elif select == 4:
            a.menu4()
        elif select == 5:
            x = a.menu5()
            print("Tỉ lệ chính xác: ", x)
        elif select == 6:
            a.menu6()
        elif select == 7:
            a.menu7()
        elif select == 8:
            a.menu8()
        elif select == 9:
            arr = {
                0: 'neutral',
                1: 'positive',
                2: 'negative'
            }
            while True:
                try:
                    name = input("Nhập text cảm xúc : ")
                    if name == '':
                        raise ValueError
                except ValueError:
                    print('Không để trống đoạn text')
                else:
                    break
            while True:
                try:
                    n = int(
                        input("[0:'neutral',1:'positive',2:'negative']=  "))
                    if n > 2 or n < 0:
                        raise ValueError
                except:
                    print('Yêu cầu nhập số 0, 1 hoặc 2')
                else:
                    break
            Test.a.Text_Speed_Sentences(name, arr[n])
        elif select == 10:
            df = Test.a.Text()  # người dùng nhập dư liệu từ bàn phím (cần xữ lí try catch khi người dùng nhập sai hoặc ràng buộc)
            # đưa dữ liệu vào và bắt đầu test xuất ra kq
            all = Test.a.Test_Model(df)
            Test.a.Result(df, all)
        elif select == 11:
            df, link = Test.a.Text_CSV()
            all = Test.a.Test_Model(df)
            Test.a.Result_CSV(df, all, link)
        elif select == 12:
            cau = input("Nhập vào chữ bạn muốn in ra: \n\n")
            a.menu12(cau)
            break
        else:
            break
