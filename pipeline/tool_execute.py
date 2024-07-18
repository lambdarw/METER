import base64
import ast
from pipeline.openai_wrapper import *
# from pipeline.tool.detect import *
from pipeline.tool.detect_yolo import *
from pipeline.tool.ocr import *
from pipeline.tool.paddle_ocr import *
from pipeline.tool.google_serper import *
from pipeline.gemini_wrapper import *

class Tool:
    def __init__(self, use_model, config):
        with open(config["prompts"]["query_generate"],"r",encoding='utf-8') as file:
            self.prompt = yaml.load(file, yaml.FullLoader)
        self.config = config
        self.detector = DetectModel(config=self.config)
        self.ocr = OCRModel(config=self.config)
        self.pp_ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu= False)
        if use_model == 'Gemini':  # gemini-1.5-flash
            self.visionchat = SyncGeminiChat(model="gemini-1.5-flash", config=self.config["gemini"])
            self.topicchat = SyncGeminiChat(model="gemini-1.5-flash", config=self.config["gemini"])
            print("执行Gemini-Tools")
        else:
            self.visionchat = SyncChat(model="gpt-4o", config=self.config["openai"])
            self.topicchat = SyncChat(model="gpt-4o", config=self.config["openai"])
            print("执行Chatgpt-4o-Tools")
        self.search = GoogleSerperAPIWrapper(config=self.config)
        
    def get_object_res(self, image_path, objects):
        object_res = self.detector.execute(image_path=image_path,
                                           content=objects,
                                           box_threshold=self.config["tool"]["detect"]["BOX_TRESHOLD"], 
                                           text_threshold=self.config["tool"]["detect"]["TEXT_TRESHOLD"],
                                           save_path='/root/zrw/code/EasyDetect-main/runs/GDetect/') 
        return object_res
        
    def get_ocr_res(self, image_path, scenetext_list):
        use_ocr = False
        for key in scenetext_list:
            if scenetext_list[key][0] != "none":
                use_ocr = True  
        ocr_res = None
        
        if use_ocr:  
            ocr_res = self.detector.execute(image_path=image_path,
                                             content=self.config["tool"]["ocr"]["content"],
                                             box_threshold=self.config["tool"]["ocr"]["BOX_TRESHOLD"], 
                                             text_threshold=self.config["tool"]["ocr"]["TEXT_TRESHOLD"],
                                             save_path=self.config["tool"]["ocr"]["cachefiles_path"])
            
            ocr_res["phrases"] = self.ocr.execute(image_path_list=ocr_res["save_path"])
            del ocr_res["save_path"]
            
            pp_ocr_res = pp_ocr(image_path, self.pp_ocr)

            merged_res = {'boxes': list(ocr_res['boxes']), 'phrases': list(ocr_res['phrases'])}

            existing_phrases = set(ocr_res['phrases'])

            for box, phrase in zip(pp_ocr_res['boxes'], pp_ocr_res['phrases']):
                if phrase not in existing_phrases:
                    merged_res['boxes'].append(box)
                    merged_res['phrases'].append(phrase)

            print("\ndetect scenetext: {}".format(merged_res))
            return merged_res
        return ocr_res
    
    def get_attribute_res(self, image_path, attribute_list):
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        queries = ""
        cnt = 1
        for key in attribute_list:
            if attribute_list[key][0] != "none":
                for query in attribute_list[key]:
                    queries += str(cnt) + "." + query + "\n"
                    cnt += 1
        if queries == "":
            attribue_res = "none information"
        else:
            img = encode_image(image_path)
            message = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": queries
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img
                            }
                        }
                    ],
                }
            ]
            # message=[
            #             {
            #                 "role": "user",
            #                 "content": [
            #                     {"type": "text", "text": queries},
            #                     {
            #                         "type": "image_url",
            #                         "image_url": {"url": f"data:image/jpeg;base64,{img}",}
            #                     },
            #                 ],
            #             }
            #         ]
            
            attribue_res = self.visionchat.get_response(message=message)
            print("attribue_res: {}\n".format(attribue_res))
        return attribue_res
    
    def get_fact_res(self, text, type, fact_list):
        '''
        type：类型 image-to-text  text-to-image
        text: 图片对应的文本描述 (数据集的字段是response)
        topic: 文本对应的领域
        fact_list: LLM 生成的提问
        fact_res: Google检索的答案
        '''   
        # 首先判断是否需要检索
        print("fact_list: {}\n".format(fact_list))
        
        if all(value == 'none' for values in fact_list.values() for value in values):
            fact_res = "none information"
        else:
            # 判断文本所属的领域topic
            self.type = type
            message = [
                {"role": "model", "parts": [{"text": self.prompt[type]["topic"]["system"]}]},
                {"role": "user", "parts": [{"text": self.prompt[type]["topic"]["user"].format(text=text)}]},
            ]
            # message = [
            #             {"role": "system", "content": self.prompt[type]["topic"]["system"]}, 
            #             {"role": "user", "content": self.prompt[type]["topic"]["user"].format(text=text)}
            #         ]
            topic_str = self.topicchat.get_response(message=message)  # ["topic"]
            try:
                topic = ast.literal_eval(topic_str)[0]
            except ValueError:
                topic = 'none'
                print("catch topic error!")
            # Google Serper
            fact_res = ""
            cnt = 1
            for key in fact_list:
                if fact_list[key][0] != "none": 
                    # 判断是否有topic
                    if topic != "none":
                        content = [str(item) + ' in the field of ' + topic for item in fact_list[key]]
                        evidences = self.search.execute(content=str(content))
                    else:
                        evidences = self.search.execute(content=str(fact_list[key]))
                    for evidence in evidences:
                        fact_res += str(cnt) + "." + evidence + "\n"
                        cnt += 1
                        
        return fact_res
    
    def execute(self, text, type, image_path, objects, attribute_list, scenetext_list, fact_list):
        object_res = self.get_object_res(image_path, objects)
        attribue_res = self.get_attribute_res(image_path, attribute_list)
        ocr_res = self.get_ocr_res(image_path, scenetext_list)
        fact_res = self.get_fact_res(text, type, fact_list)
        return object_res, attribue_res, ocr_res, fact_res



            
            
            
   