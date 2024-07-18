import yaml
import json
from pipeline.openai_wrapper import *
from pipeline.claim_generate import * 
from pipeline.query_generate import *
from pipeline.tool_execute import *
from pipeline.verify import *
from pipeline.gemini_wrapper import *


class Pipeline:
    def __init__(self, use_model):
        with open("pipeline/config/config.yaml", 'r', encoding='utf-8') as file:
            self.config = yaml.load(file, yaml.FullLoader)
        if use_model == 'Gemini':  # gemini-1.5-flash
            self.syncchat = SyncGeminiChat(model="gemini-1.5-flash", config=self.config["gemini"])
            self.asyncchat = AsyncGeminiChat(model="gemini-1.5-flash", config=self.config["gemini"])
            self.visionchat = SyncGeminiChat(model="gemini-1.5-flash", config=self.config["gemini"])
        else:
            self.syncchat = SyncChat(model="gpt-4o", config=self.config["openai"])
            self.asyncchat = AsyncChat(model="gpt-4o", config=self.config["openai"])
            self.visionchat = SyncChat(model="gpt-4o", config=self.config["openai"])

        self.claim_generator = ClaimGenerator(config=self.config,chat=self.syncchat)
        self.query_generator = QueryGenerator(config=self.config,chat=self.asyncchat)
        self.tool = Tool(use_model=use_model, config=self.config)
        self.verifier = Verifier(config=self.config, chat=self.visionchat)

    
    def run(self, text, image_path, type):
        claim_list = text
        
        
        objects, attribute_ques_list, scenetext_ques_list, fact_ques_list = self.query_generator.get_response(claim_list=claim_list, type=type)
        object_res, attribue_res, text_res, fact_res = self.tool.execute(text=text, 
                                                                        type=type, 
                                                                        image_path=image_path,
                                                                        objects=objects,
                                                                        attribute_list=attribute_ques_list, 
                                                                        scenetext_list=scenetext_ques_list, 
                                                                        fact_list=fact_ques_list)
        response = self.verifier.get_response(type, object_res, attribue_res, text_res, fact_res, claim_list, image_path)
        return response, claim_list
        

