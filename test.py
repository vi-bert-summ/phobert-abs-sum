import os
import glob
import time
import datasets
import pandas as pd
import transformers
import concurrent.futures
from typing import Optional

import utils
import parameters

# from datasets import *
from vncorenlp import VnCoreNLP
from transformers import EncoderDecoderModel
from transformers import AutoTokenizer

args = parameters.get_args()


if __name__ == '__main__':
    start = time.time()
    text = '''
 Ngày 4/2, lãnh đạo thị xã Hoàng Mai (Nghệ An) cho biết, cơ quan chức năng thị xã vừa phối hợp với đơn vị chức năng tỉnh cách ly, lấy mẫu xét nghiệm cho 1 nam sinh viên trường Đại học FPT có biểu hiện ho sốt sau khi đi từ Hà Nội về nhà. Trước đó vào ngày 31/1 nam sinh C.T.Ph. (20 tuổi, trú TX. Hoàng Mai, Nghệ An; Sinh viên trường Đại học FPT cơ sở Mỹ Đình) cùng 1 người bạn đi xe máy từ Hà Nội về quê nhà.
Lúc 22h ngày 31/1 nam sinh này về đến quê nhà nhưng không khai báo y tế. Từ ngày 1-3/2, nam sinh Ph. có đi một số nơi trên địa bàn thị xã và tiếp xúc với khoảng 12 người. Khoảng 20h ngày 3/2, nam sinh này có dấu hiệu sốt 39 độ, đau đầu, đau rát họng, đau mỏi cơ, khó thở nhẹ. Nam sinh này sau đó được đưa vào Bệnh viện Phong da liễu (TX. Hoàng Mai) để khám và cách ly y tế.
Nhận được tin báo, Trung tâm Kiểm soát bệnh tật tỉnh Nghệ An đã trực tiếp ra lấy mẫu bệnh phẩm để xét nghiệm Covid-19. Dự kiến trong chiều cùng ngày sẽ có kết quả. Hiện cơ quan chức năng đang rà soát những người tiếp xúc gần với nam sinh Ph. để có phương án theo dõi, cách ly và lấy mẫu xét nghiệm. Trước đó vào đêm 31/1, một chuyến xe khách chở 22 người từ Hà Nội về Nghệ An trong đó có nhiều sinh viên Đại học FPT đi về tại phường Lê Lợi, TP. Vinh (Nghệ An). Cơ quan chức năng sau đó đã đưa 1 trường hợp F1 trên xe khách này đi cách ly tại Trung tâm Y tế huyện Nam Đàn. Riêng những trường hợp còn lại được theo dõi tại nhà. Sau khi cách ly, các trường hợp này đã được lấy mẫu xét nghiệm và cho kết quả âm tính với Covid-19.
 '''
    rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g') 

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    output = utils.bertsum(text, tokenizer, rdrsegmenter, args.device, args.checkpoint)

    print(output)
    print('Time: ', time.time() - start)