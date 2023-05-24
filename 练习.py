# 3.2 Word2Vec
# # 导入模块，读取搜狐数据
# # -* coding: utf-8 *-
# import jieba
# from gensim.models import Word2Vec
# import pickle
# import numpy as np
#
# # 1.导入搜狐数据，对数据进行处理
# S1 = '序号时间活动内容活动现场前期布展客户展厅集合签到签试驾协议试驾分组领取胸卡统计客户买统一乘坐大巴活动现场媒体签到客户签到移至篷房区休息景逸宣传片播放劲舞开场主持人开场介绍来宾活动主题经销商领导讲话产品讲解试驾流程介绍试驾车手表演排序试驾试乘等待人员篷房休息午餐区用餐自由活动客户拍照发微薄询价购车礼品发放返回店内购车环节'
# S2 = '搜狐汽车沈阳站近日编辑长安汽车辽宁华威店款奔奔手动亲情版时尚版优惠 元现金店内现车颜色全奔奔手动亲情版时尚版外车型暂无现车接受预订咨询电话价格 信息下表注上图资料图片信息无关动力奔奔采用四缸气门双顶置凸轮轴发动机排量功率扭矩配合档手动自动变速箱保养长安奔奔整车质保周期厂家提供年万公里奔奔日常 保养周期公里保养价格元保养车况而定编辑建议提到奔奔年轻人想起动画片里说话小汽车长安奔奔可爱灵动形象打动消费者时尚外观精致内饰价格一款性价比不错代步车 适合女性消费者该车购车可享优惠'
# # 2.自定义读取数据类
# class readdata():
#     def __init__(self,contexts):
#         self.contexts = contexts
# # 3.构建词汇表
#     def __iter__(self):
#         for line in self.contexts:
#             yield list(jieba.cut(line))
#
# sentences = [S1,S2]
# contexts = readdata(sentences)
# print(contexts)
# # 4.参数选择：窗口大小选择5，此向量为度选择300，使用CBOW+树进行训练
# # model = Word2Vec(
# #     contexts,
# #     vector_size=300,
# #     sg=0,#sg:skip-gram or cbow, 0为cbow
# #     cbow_mean=1,# cbow_mean: 如果为0,则采用上下文词向量的和,如果为1(default)则采用均值;
# #     hs=1 # hs: 如果是0,则是Negative Sampling,是1的话并且负采样个数negative大于0,则是Hierarchical Softmax
# #     # window=5,# window=5 默认
# # )
# # 5.将模型保存
# # model1 = open('model_word.pk','wb')
# # pickle.dump(model,model1)
# # model1.close()
# # model.save('model_word.pk')
# # 6.模型读取，并加载wiki数据
# # model2 = open('model_word.pk','rb')
# # model_load = pickle.load(model2)
# # model2.close()
# model = Word2Vec.load('model_word.pk')
# # # 7.正确完成增量训练
# # # 8.将s1和s2句子进行分词的处理
# S1_jieba = list(jieba.cut(S1))
# S2_jieba = list(jieba.cut(S2))
# # 9.将分词之后的s1，s2每个词通过model编码成句子向量
# s1_vec = [model.wv.get_vector(word) for word in S1_jieba if word in model.wv.index_to_key]
# s2_vec = [model.wv.get_vector(word) for word in S2_jieba if word in model.wv.index_to_key]
# # 10.将s1向量和s2向量进行相似度的计算
# similary = np.corrcoef(s1_vec,s2_vec)
# # 11.打印相似值
# print(similary)


