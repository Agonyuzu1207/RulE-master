import sys
import os
import logging
import argparse
import random
import json
import yaml
import easydict
import numpy as np
import torch
import jinja2

def load_config(cfg_file):#读取参数
    # cfg_file = os.path.join(cfg_file, 'config.json')
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()

    if "---" in raw_text:
        configs = []
        grid, template = raw_text.split("---")
        grid = yaml.safe_load(grid)
        template = jinja2.Template(template)
        for hyperparam in np.meshgrid(grid):
            config = easydict.EasyDict(yaml.safe_load(template.render(hyperparam)))
            configs.append(config)
    else:
        configs = [easydict.EasyDict(yaml.safe_load(raw_text))]

    return configs

#保存参数
def save_config(args):

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

def set_seed(seed):#800，生成随机数
    torch.manual_seed(seed)#设置CPU生成随机数的种子，返回一个torch.Generator对象
    random.seed(seed)#random.seed() 会改变随机生成器的种子；如果使用相同的seed()值，则每次生成的随机数都相同
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_model(model, optim, args):

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    params = {
        'model': model.state_dict(),
        'optim': optim.state_dict()
    }

    torch.save(params, os.path.join(args.save_path, 'checkpoint'))
    
    entity_embedding = model.entity_embedding.weight.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.weight.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )
    
    rule_embedding = model.rule_emb.weight.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'rule_embedding'), 
        rule_embedding
    )

def load_model(model, optim, args):
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])

#设置日志
def set_logger(save_path):
    log_file = os.path.join(save_path, 'run.log')#日志文件路径

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()#将日志输出到屏幕上
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')#时间，日志记录级别。消息
    console.setFormatter(formatter)#设置格式
    logging.getLogger('').addHandler(console)##为root logger添加handler