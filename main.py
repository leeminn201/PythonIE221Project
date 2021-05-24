pip install transformers
!pip3 install sentencepiece
!pip3 install tf_sentencepiece


import sys
sys.path.insert(0,'/content/drive/MyDrive/Kỹ thuật lập trình Python/Code_Python/Code_Luong')

from Library import library
from load_data import load

from Library import library as l
from load_data import load

from tranning import train
from VisualData.visual_data import Visual
from tranning import Test_cau



class Menu():
    def __init__(self):
        pass

    def menu1(self):
        return load.train.shape(), load.test.shape()

    def menu2(self):
        return load.train.head(30)

    def menu3(self):
        return load.test.head(30)

    def menu4(self):
        x = b.Model1(1, 96)
        return x.build_model()

    def menu5(self):
        Build = Train(MAX_LEN, PATH, p.a.input_ids, p.a.input_ids_t, p.a.attention_mask,
                p.a.attention_mask_t, p.a.token_type_ids, p.a.token_type_ids_t, p.a.start_tokens, p.a.end_tokens)
        return Build.trainModel()

    def menu6(self, model):
        Build = Train(load.MAX_LEN, load.PATH, load.a.input_ids, load.a.input_ids_t, load.a.attention_mask,
                     load.a.attention_mask_t, load.a.token_type_ids, load.a.token_type_ids_t, load.a.start_tokens, load.a.end_tokens)
        return Build.accuracy()

    def menu7():
        return Visual.visualTrain()

    def menu8():
        return Visual.visualTest()

    def menu9():
        return Visual.all_of_Train()

    def menu10():
        return Visual.wordCloud()

    def menu11():
        return Visual.histogram()

    def menu12():
        statistic = Visual.all_of_train()
        return stattistic.describe()

    def menu13():
        name = input("Nhập text cảm xúc : ")
        arr = {
            0: 'neutral',
            1: 'positive',
             2: 'negative'
         }
        n = int(input("[0:'neutral',1:'positive',2:'negative']=  "))
        return Test.a.Text_speed_1cau(name,arr[n])  

    def menu14():
        #Test với 1 hoặc nhiều câu lưu vô file D:\UIT LEARN\Năm 3 Kì 2\Python\do_an\doAN\Dataset\submission.csv đây là file chính lưu dữ liệu người dùng đưa vào
        df=Test.a.TEXT()  #người dùng nhập dư liệu từ bàn phím (cần xữ lí try catch khi người dùng nhập sai hoặc ràng buộc)
        all=Test.a.TEST_MODEL(df)  # đưa dữ liệu vào và bắt đầu test xuất ra kq
        return Test.a.KQ(df,all)

    def menu15():
        #Test với 1 file csv bất kì nhung phải có header là "text", "sentiment" sai định dạng cút
        df,link=Test.a.TEXT_CSV()
        all=Test.a.TEST_MODEL(df)
        return Test.a.KQ_ADD_CSV(df,all,link)

while True:
    print('===============================MENU===============================')
    print('------------< Mời bạn chọn tính năng với số tương ứng >-----------')
    print('||                                                               ||')
    print('||   1.  Load dữ liệu lên                                        ||')
    print('||   2.  Xem kích thước tập train, tập test                      ||')
    print('||   3.  Xem bộ dữ liệu train                                    ||')
    print('||   4.  Xem bộ dữ liệu test                                     ||')
    print('||   5.  Xây dựng mô hình                                        ||')
    print('||   6.  Đánh giá độ chính xác                                   ||')
    print('||   7.  Kết quả dự đoán trên tập test                           ||')
    print('||   8.  Kết quả dự đoán trên tập train                          ||')
    print('||   9.  Trực quan hóa thành Word Cloud                          ||')
    print('||   10. Trực quan hóa tỉ lệ dự đoán                             ||')
    print('||   11. Thống kê bộ dữ liệu train                               ||')
    print('||   12. Chạy thử trên 1 câu tự nhập                             ||')
    print('||   13. Chạy thử trên một bộ DataFrame                          ||')
    print('||   14. Lưu kết quả dưới dạng file csv                          ||')
    print('||   15. Kết thúc chương trình                                   ||')
    print('||                                                               ||')
    print('---------< Nhập ký tự bất kỳ để kết thúc chương trình >-----------')
    print('============================THE END===============================')
    try:
        select = int(input())
    except:
        pass
    if select == 1:
        menu1()
    elif select == 2:
        menu2()
    elif select == 3:
        menu3()
    elif select == 4:
        menu4()
    elif select == 5:
        menu5()
    elif select == 6:
        menu6()
    elif select == 7:
        menu7()
    elif select == 8:
        menu8()
    elif select == 9:
        menu9()
    elif select == 10:
        menu10()
    elif select == 11:
        menu11()
    elif select == 12:
        menu12()
    elif select == 13:
        menu13()
    elif select == 14:
        menu14()
    elif select == 15:
        menu15()
    else:
        break

