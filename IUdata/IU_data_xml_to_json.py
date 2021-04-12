from lxml import etree
import json
import os
from collections import Counter
import random


def xml_read(f):
    tree = etree.parse(f)
    findings = tree.find(".//AbstractText[@Label='FINDINGS']")
    impression = tree.find(".//AbstractText[@Label='IMPRESSION']")
    parentimages = tree.findall(".//parentImage")
    x = [findings, impression]
    y = [i.text for i in x]
    findings_txt = y[0]
    impressions_txt = y[1]
    parentimgs_lst = [i.attrib['id'] for i in parentimages]
    return parentimgs_lst, impressions_txt, findings_txt


def dict_save_json(dict, file_name):
    with open('{}'.format(file_name), 'w') as file_position:
        json.dump(dict, file_position)
    print('save json file: {}'.format(file_name))


"""Generate the .json data to fit Xue, 2018 paper. I have already done it for you
    If you wanna generate you own,  you need to download xml and png data from IU X-ray dataset website"""

if __name__ == '__main__':
    xml_path = '../../datasets_origin/ecgen-radiology'  # you need to download .xml data from IU X-ray dataset website
    frontal_image_path = '../IUdata/NLMXCR_Frontal'
    # The image name
    frontal_names_lst = []
    # The image number after CXR
    frontal_num_lst = []
    # one_sentence_impression = False
    json_dict = {}
    json_dict_train = {}
    json_dict_test = {}

    # The image dataset with only frontal views is selected manually.
    for frontal_image_name in os.listdir(frontal_image_path):
        frontal_names_lst.append(str(frontal_image_name).split(".")[0])

    for name in os.listdir(xml_path):
        xml_file = os.path.join(xml_path, name)
        parentimgs_lst, impression, finding = xml_read(xml_file)
        # must have impression and finding
        if impression and finding:
            impression = str(impression)
            finding = str(finding)

            # choose the frontal image only for its related impression and finding
            for image_name in parentimgs_lst:
                image_name = str(image_name)
                for frontal_img_name in frontal_names_lst:
                    if image_name == frontal_img_name:
                        assert type(frontal_img_name) == str
                        json_dict[frontal_img_name] = [impression, finding]
                        if random.random() < 0.3 and len(json_dict_test.keys()) < 300:
                            json_dict_test[frontal_img_name] = [impression, finding]
                        else:
                            json_dict_train[frontal_img_name] = [impression, finding]

    if len(json_dict_test.keys()) != 300:
        print("Please run this program AGAIN!")
    assert len(json_dict_test.keys()) == 300

    print('The number of complete radiology reports with the frontal view in total:', len(json_dict.keys()))
    print('The number of complete radiology reports with the frontal view in training set:', len(json_dict_train.keys()))
    print('The number of complete radiology reports with the frontal view in testing set:', len(json_dict_test.keys()))

    # gnerate json file from the dictionary
    dict_save_json(json_dict, "IUdata_trainval.json")
    dict_save_json(json_dict_train, "IUdata_train.json")
    dict_save_json(json_dict_test, "IUdata_test.json")
