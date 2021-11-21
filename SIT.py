import hashlib
import json
import os.path
import random
import string
import time
import uuid
from hashlib import md5
import jieba
import nltk
import requests
import torch
from googletrans import Translator
from nltk import CoreNLPDependencyParser
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM


def pretreatment(sent):
    '''
    :param sent: 传入字符串
    :return: 返回去除句点并替换缩写后的字符串
    '''
    sent = sent[:-1]  # 去除句点，便于后续处理
    sent = sent.replace("\'re ", " are ")   # 替换常见缩写，下同
    sent = sent.replace("\'m ", " am ")
    # sent = sent.replace("\'s ", " is ")
    return sent


def perturb(sent, bertmodel, num):
    '''
    :param sent: 输入的待翻译语句
    :param bertmodel: bert模型
    :param num: 每个名词或形容词的近义词替代数
    :return: 待翻译语句扩增后的列表
    '''
    sent = pretreatment(sent)
    tokens = tokenizer.tokenize(sent)   # 分词
    pos_inf = nltk.tag.pos_tag(tokens)  # 词性
    # 列表中元素：(该词的索引，该词的词性)
    bert_masked_indexL = list()
    # 计算对应索引的替换词
    for idx, (word, tag) in enumerate(pos_inf):
        # 替换名词和形容词
        if (tag.startswith('NN') or tag.startswith('JJ')):
            tagFlag = tag[:2]   # 规范词性为NN/JJ
            # 不替换第一个和最后一个词，Bert在这两个位置的表现相对较差
            if (idx != 0 and idx != len(tokens) - 1):
                bert_masked_indexL.append((idx, tagFlag))
    # 使用Bert生成相似语句
    bert_new_sentences = list()
    if bert_masked_indexL:
        bert_new_sentences = perturbBert(sent, bertmodel, num, bert_masked_indexL)
    return bert_new_sentences


def perturbBert(sent, bertmodel, num, masked_indexL):
    '''
    :param sent: 输入的待翻译语句
    :param bertmodel: bert模型
    :param num: 每个名词或形容词的近义词替代数
    :param masked_indexL: 该语句分词列表中需要被替换的词
    :return: 待翻译语句扩增后的列表
    '''
    new_sentences = list()
    tokens = tokenizer.tokenize(sent)
    tokens = [x.lower() for x in tokens]

    invalidChars = set(string.punctuation)  # 标点
    for (masked_index, tagFlag) in masked_indexL:   # 依次遮罩需要更换的词汇
        original_word = tokens[masked_index]
        cur_tokens = tokens
        cur_tokens[masked_index] = '[MASK]'

        try:
            indexed_tokens = berttokenizer.convert_tokens_to_ids(cur_tokens)    # 转换tokens为词表中的id
            tokens_tensor = torch.tensor([indexed_tokens])  # 转换为张量
            prediction = bertmodel(tokens_tensor)   # 预测
        # skip the sentences that contain unknown words, we skip sentences to reduce fp caused by BERT
        # TODO another option is to mark the unknown words as [MASK];
        except KeyError as error:
            print('skip a sentence. unknown token is %s' % error)
            break

        # 获取num个最相似词汇
        topk_Idx = torch.topk(prediction[0, masked_index], num)[1].tolist() # prediction[0, masked_index]取第masked_index行
        topk_tokens = berttokenizer.convert_ids_to_tokens(topk_Idx) # 转换词表中id为tokens

        # 去除长度不大于1的无意义字母标点
        # TODO this step could be further optimized by filtering more tokens (e.g., non-English tokens)
        topk_tokens = list(filter(lambda x: len(x) > 1, topk_tokens))

        # 生成扩增语句
        for t in topk_tokens:
            if any(char in invalidChars for char in t): # 相似词不为标点不包含标点
                continue
            tokens[masked_index] = t
            new_pos_inf = nltk.tag.pos_tag(tokens)
            # 仅保留词性同原词相同的预测词作为扩增项
            if (new_pos_inf[masked_index][1].startswith(tagFlag)):
                new_sentence = detokenizer.detokenize(tokens)
                new_sentences.append(new_sentence)
        tokens[masked_index] = original_word    # 还原

    return new_sentences


def bing_translate(input):
    # Add your subscription key and endpoint
    subscription_key = "出于安全考虑就删掉了"
    endpoint = "https://api.cognitive.microsofttranslator.com"
    # Add your location, also known as region. The default is global.
    # This is required if using a Cognitive Services resource.
    location = "global"
    path = '/translate'
    constructed_url = endpoint + path
    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': 'zh-Hans'
    }
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    res = list()
    for index in range(len(input)):
        body = [{"text": input[index]}]
        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        response = request.json()
        res.append([x['translations'][0]['text'] for x in response][0])
    return res


def google_translate(input):
    translator = Translator(service_urls=['translate.google.cn', ])
    res = []
    for origin in input:
        trans = translator.translate(origin, src='en', dest='zh-cn')
        res.append(trans.text)
    return res


def baidu_translate(input):
    res = list()
    for query in input:
        res.append(execute_baidu_translate(query))
    return res


def execute_baidu_translate(query):
    # Set necessary infos for baidu
    appid = '同样出于安全考虑删除了'
    appkey = '同样出于安全考虑删除了'
    from_lang = 'en'
    to_lang = 'zh'
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    # Generate salt and sign
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build and send request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    return result['trans_result'][0]['dst']


def youdao_translate(input):
    res = list()
    for query in input:
        res.append(execute_youdao_translate(query))
    return res


def execute_youdao_translate(query):
    YOUDAO_URL = 'https://openapi.youdao.com/api'
    APP_KEY = '同样出于安全考虑删除了'
    APP_SECRET = '同样出于安全考虑删除了'

    data = {}
    data['from'] = 'en'
    data['to'] = 'zh-CHS'
    data['signType'] = 'v3'
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())

    def truncate(q):
        if q is None:
            return None
        size = len(q)
        return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

    signStr = APP_KEY + truncate(query) + salt + curtime + APP_SECRET
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    sign = hash_algorithm.hexdigest()
    data['appKey'] = APP_KEY
    data['q'] = query
    data['salt'] = salt
    data['sign'] = sign
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(YOUDAO_URL, data=data, headers=headers).content
    response = json.loads(response.decode())
    try:
        res = response["translation"][0]
    except:
        print(response)
        res = ""
    return res


def execute_translate(input, software):
    '''
    :param input: 待翻译语句列表
    :param software: 指定翻译软件
    :return: 翻译结果列表及分词后列表
    '''
    target_sentsL = []
    target_sentsL_seg = []
    if software == "google":
        target_sentsL = google_translate(input)
    elif software == "bing":
        target_sentsL = bing_translate(input)
    elif software == "baidu":
        target_sentsL = baidu_translate(input)
    elif software == "youdao":
        target_sentsL = youdao_translate(input)
    else:
        print("check your software input!")
        exit(-1)
    for i in target_sentsL:
        cur_seg = ' '.join(jieba.cut(i))
        target_sentsL_seg.append(cur_seg)
    return target_sentsL, target_sentsL_seg


# 计算原始语句翻译和扩增语句翻译后的语法依赖树距离
def depDistance(graph1, graph2):
    counts1 = dict()
    for i in graph1:
        counts1[i[1]] = counts1.get(i[1], 0) + 1
    counts2 = dict()
    for i in graph2:
        counts2[i[1]] = counts2.get(i[1], 0) + 1
    all_deps = set(list(counts1.keys()) + list(counts2.keys()))
    diffs = 0
    for dep in all_deps:
        diffs += abs(counts1.get(dep, 0) - counts2.get(dep, 0))
    return diffs


if __name__ == "__main__":
    # todo  to set values
    dataset = "politics"    # 数据集
    software = "google"  # 翻译软件
    num_perturb = 10    # 单句扩增数目
    is_perturb = False
    perturb_all = list()
    bertmodel = None
    distance_threshold = 0.0  # 语法树结构比较时差异的阈值，超过则记为翻译错误
    issue_threshold = 3  # 单句输出问题个数
    issue_count = 0
    output_file = './results/results_' + dataset + '_' + software + '.txt'  # 结果输出
    write_output = open(output_file, 'w', encoding='utf-8')

    # 尝试加载扩增语句
    if os.path.exists("./store_results/res_perturb/perturb_{}.json".format(dataset)):
        is_perturb = True
        with open('./store_results/res_perturb/perturb_{}.json'.format(dataset), 'r', encoding="utf-8") as f:
            perturb_all = json.load(f)
            f.close()
        print("Successfully load perturbed data!")
    # initialize the dependency parser
    chi_parser = CoreNLPDependencyParser('http://localhost:9001')

    # use nltk treebank tokenizer and detokenizer
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    print("nltk initialized!")

    if not is_perturb:
        # BERT initialization
        # bert-large-uncased: 24-layer, 1024-hidden, 16-heads, 340M parameters
        berttokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        bertmodel = BertForMaskedLM.from_pretrained('bert-large-uncased')
        bertmodel.eval()
        print("Bert initialized!")

    # load source sentences
    origin_source_sentsL = []
    with open('./dataset/' + dataset, encoding='utf-8') as file:
        for line in file:
            origin_source_sentsL.append(line.strip())
        file.close()
    print("Successfully load dataset!")

    origin_target_sentsL = list()
    origin_target_sentsL_seg = list()
    # 尝试加载原始语句的翻译及分词结果，否则新生成并存储到本地
    if os.path.exists("./store_results/trans/{}/{}_origin.json".format(software, dataset)):
        with open("./store_results/trans/{}/{}_origin.json".format(software, dataset), 'r', encoding="utf-8") as f:
            origin_trans_data = json.load(f)
            f.close()
        for x in origin_trans_data:
            origin_target_sentsL.append(x["target"])
            origin_target_sentsL_seg.append(x["segments"])
    else:
        origin_target_sentsL, origin_target_sentsL_seg = execute_translate(input=origin_source_sentsL, software=software)
        origin_trans_data = list()
        for i in range(len(origin_source_sentsL)):
            origin_trans_data.append({"index":i,
                                      "target":origin_target_sentsL[i],
                                      "segments":origin_target_sentsL_seg[i]})
        origin_trans_data = json.dumps(origin_trans_data, ensure_ascii=False, indent=4)
        with open(r'./store_results/trans/{}/{}_origin.json'.format(software, dataset), 'w', encoding='utf-8') as f:
            f.write(origin_trans_data)
    print("Input sentences translated!")
    # 解析分词后的原始输入语句译文，生成语法依赖树
    origin_target_treesL = [i for (i,) in chi_parser.raw_parse_sents(origin_target_sentsL_seg, properties={'ssplit.eolonly': 'true'})]
    print("Input sentences parsed!")

    # 尝试加载扩增语句的翻译及分词结果
    new_target_sentsL_seg_all = list()
    new_target_sentsL_all = list()
    if is_perturb and os.path.exists('./store_results/trans/{}/{}_new_sentences.json'.format(software, dataset)):
        with open('./store_results/trans/{}/{}_new_sentences.json'.format(software, dataset), 'r', encoding="utf-8") as f:
            new_trans_data = json.load(f)
            f.close()
        for x in new_trans_data:
            new_target_sentsL_seg_all.append(x["segments"])
            new_target_sentsL_all.append(x["target"])
    else:
        new_trans_data = list()
        for idx, origin_source_sent in enumerate(origin_source_sentsL):
            # 扩增语句并翻译，保存结果
            if not is_perturb:
                new_source_sentsL = perturb(origin_source_sent, bertmodel, num_perturb)
                perturb_all.append({origin_source_sent: new_source_sentsL})
            else:
                new_source_sentsL = perturb_all[idx][origin_source_sent]
            if len(new_source_sentsL) == 0:
                new_trans_data.append({"index":idx,
                                       "target":list(),
                                       "segments":list()})
                new_target_sentsL_seg_all.append(list())
                new_target_sentsL_all.append(list())
                continue
            new_target_sentsL, new_target_sentsL_seg = execute_translate(input=new_source_sentsL, software=software)
            new_trans_data.append({"index":idx,
                                   "target":new_target_sentsL,
                                   "segments":new_target_sentsL_seg})
            new_target_sentsL_seg_all.append(new_target_sentsL_seg)
            new_target_sentsL_all.append(new_target_sentsL)
            print(idx, origin_source_sent)
            print('number of sentences: ', len(new_source_sentsL))
        perturb_all_data = json.dumps(perturb_all, ensure_ascii=False, indent=4)
        with open(r'./store_results/res_perturb/perturb_{}.json'.format(dataset), 'w', encoding='utf-8') as f:
            f.write(perturb_all_data)
        new_trans_data = json.dumps(new_trans_data, ensure_ascii=False, indent=4)
        with open(r'./store_results/trans/{}/{}_new_sentences.json'.format(software, dataset), 'w', encoding='utf-8') as f:
            f.write(new_trans_data)

    for idx, origin_source_sent in enumerate(origin_source_sentsL):
        new_target_sentsL_seg = new_target_sentsL_seg_all[idx]
        new_source_sentsL = perturb_all[idx][origin_source_sent]
        new_target_sentsL = new_target_sentsL_all[idx]
        origin_target_tree = origin_target_treesL[idx]
        origin_target_sent = origin_target_sentsL[idx]
        suspicious_issues = list()
        if len(new_source_sentsL) == 0:
            continue
        # 获取扩增语句的语法依赖树
        new_target_treesL = [target_tree for (target_tree,) in chi_parser.raw_parse_sents(new_target_sentsL_seg, properties={'ssplit.eolonly': 'true'})]
        assert (len(new_target_treesL) == len(new_source_sentsL))
        print('new target sentences parsed')
        # 计算距离
        for (new_source_sent, new_target_sent, new_target_tree) in zip(new_source_sentsL, new_target_sentsL, new_target_treesL):
            distance = depDistance(origin_target_tree.triples(), new_target_tree.triples())
            if distance > distance_threshold:
                suspicious_issues.append((new_source_sent, new_target_sent, distance))
        print('distance calculated')

        # 按距离为键方便后续排序
        suspicious_issues_cluster = dict()
        for (new_source_sent, new_target_sent, distance) in suspicious_issues:
            if distance not in suspicious_issues_cluster:
                new_cluster = [(new_source_sent, new_target_sent)]
                suspicious_issues_cluster[distance] = new_cluster
            else:
                suspicious_issues_cluster[distance].append((new_source_sent, new_target_sent))
        print('clustered')
        # 如果无问题
        if len(suspicious_issues_cluster) == 0:
            continue
        issue_count += 1

        write_output.write(f'ID: {issue_count}\n')
        write_output.write('Source sent:\n')
        write_output.write(origin_source_sent)
        write_output.write('\nTarget sent:\n')
        write_output.write(origin_target_sent)
        write_output.write('\n\n')

        # 按distance降序排序
        sorted_keys = sorted(suspicious_issues_cluster.keys())
        sorted_keys.reverse()

        remaining_issue = issue_threshold
        # 输出前issue_threshold个问题
        for distance in sorted_keys:
            if remaining_issue == 0:
                break
            candidateL = suspicious_issues_cluster[distance]
            if len(candidateL) <= remaining_issue:
                remaining_issue -= len(candidateL)
                for candidate in candidateL:
                    write_output.write('Distance: %f\n' % (distance))
                    write_output.write(candidate[0] + '\n' + candidate[1] + '\n')
            else:
                sortedL = sorted(candidateL, key=lambda x: len(x[1]))
                issue_threshold_current = remaining_issue
                for i in range(issue_threshold_current):
                    write_output.write('Distance: %f\n' % (distance))
                    write_output.write(sortedL[i][0] + '\n' + sortedL[i][1] + '\n')
                    remaining_issue -= 1
        write_output.write('\n')
        print('result outputed')

    write_output.close()