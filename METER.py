from pipeline.run_pipeline import *
from results.evaluation import *
import ast
import os
import random
import time
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# read file
def read_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


# write json file
def write_json_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# write txt file
def write_txt_file(filename, data):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(data)
    print("write txt file success!")


def eval_result(infile, outfile):
    data = read_json_file(infile)
    
    t2i_true_labels, t2i_predicted_labels = [], []    
    i2t_true_labels, i2t_predicted_labels = [], []
    all_true_labels, all_pred_labels = [], []
    vqa_true_labels, vqa_pred_labels = [], []
    ic_true_labels, ic_pred_labels = [], []

    for item in data:
        if item["task_type"] == "text-to-image":
            t2i_true_labels.append(item["actual_label"])
            t2i_predicted_labels.append(item["predict_label"])
        else:
            i2t_true_labels.append(item["actual_label"])
            i2t_predicted_labels.append(item["predict_label"])
            if item["task_type"] == "vqa":
                vqa_true_labels.append(item["actual_label"])
                vqa_pred_labels.append(item["predict_label"])
            elif item["task_type"] == "image-caption":
                ic_true_labels.append(item["actual_label"])
                ic_pred_labels.append(item["predict_label"])

        all_true_labels.append(item["actual_label"])
        all_pred_labels.append(item["predict_label"])
    
    # evaluating
    accuracy, precision, recall, f1_score = evaluate_result(t2i_true_labels, t2i_predicted_labels)
    print("text-to-image Task - Accuracy:{}, Precision:{}, Recall:{}, F1 Score:{}".format(accuracy, precision, recall, f1_score))
    eva1 = "text-to-image Task - Accuracy:{}, Precision:{}, Recall:{}, F1 Score:{}\n".format(accuracy, precision, recall, f1_score)
    write_txt_file(outfile, data=eva1)

    accuracy, precision, recall, f1_score = evaluate_result(i2t_true_labels, i2t_predicted_labels)
    print("image-to-text Task - Accuracy:{}, Precision:{}, Recall:{}, F1 Score:{}".format(accuracy, precision, recall, f1_score))
    eva2 = "image-to-text Task - Accuracy:{}, Precision:{}, Recall:{}, F1 Score:{}\n".format(accuracy, precision, recall, f1_score)
    write_txt_file(outfile, data=eva2)

    accuracy, precision, recall, f1_score = evaluate_result(vqa_true_labels, vqa_pred_labels)
    print("vqa Task - Accuracy:{}, Precision:{}, Recall:{}, F1 Score:{}".format(accuracy, precision, recall, f1_score))
    eva3 = "vqa Task - Accuracy:{}, Precision:{}, Recall:{}, F1 Score:{}\n".format(accuracy, precision, recall, f1_score)
    write_txt_file(outfile, data=eva3)

    accuracy, precision, recall, f1_score = evaluate_result(ic_true_labels, ic_pred_labels)
    print("ic Task - Accuracy:{}, Precision:{}, Recall:{}, F1 Score:{}".format(accuracy, precision, recall, f1_score))
    eva4 = "ic Task - Accuracy:{}, Precision:{}, Recall:{}, F1 Score:{}\n".format(accuracy, precision, recall, f1_score)
    write_txt_file(outfile, data=eva4)

    accuracy, precision, recall, f1_score = evaluate_result(all_true_labels, all_pred_labels)
    print("Total Task - Accuracy:{}, Precision:{}, Recall:{}, F1 Score:{}".format(accuracy, precision, recall, f1_score))
    eva5 = "Total Task - Accuracy:{}, Precision:{}, Recall:{}, F1 Score:{}\n".format(accuracy, precision, recall, f1_score)
    write_txt_file(outfile, data=eva5)

    write_txt_file(outfile, data="***************************************\n")


# for test set
def test_examples(data_file, res_file, use_model):
    
    pipeline = Pipeline(use_model)
    results = []

    data = read_json_file(data_file)
    
    for item in data:
        res = dict()
        res['prompt'] = item['prompt']
        res['task_type'] = item['task_type']
        res['model'] = item['model']
        res['image_path'] = item['image_path']
        print("image_path:{}, type:{}\n".format(res['image_path'], res['task_type']))
        
        # processing
        count = 1
        claims_string = ""
        for claim in item["claim_list"]:
            claims_string += f"claim{count}: {claim}\n"
            count += 1

        # hullcination detecting
        if res['task_type']=='image-caption' or res['task_type']=="vqa":
            task_type = "image-to-text"
        else:
            task_type = "text-to-image"
        response_str, claim_list = pipeline.run(text=claims_string, image_path=res['image_path'], type=task_type)
        print("response_str: {}\n".format(response_str))

        response = ast.literal_eval(response_str)  # response  str->dict
        claim_list = [claim[8:] for claim in claim_list.split("\n")]
        
        # results
        res['claims'] = []
        res['predict_label'] = "non-hallucination"
        for i in range(len(response)):
            claim_num = "claim"+str(i+1)
            
            cla = dict()
            cla['claim'] = claim_list[i]
            cla['claim_label'] = response[i][claim_num]
            cla['claim_reason'] = response[i]["reason"]
            
            if response[i][claim_num] == "hallucination":
                res['predict_label'] = "hallucination" 

            res['claims'].append(cla)
        
        results.append(res)
        
        # write file
        write_json_file(res_file, data=results)


if __name__ == '__main__':
    # detecting
    use_model = "Gemini"
    data_file = # your dataset
    res_file = # your results file
    test_examples(data_file, res_file, use_model)
    
    # evaluating
    eval_file = # your evaluating file
    eval_result(res_file, eval_file)