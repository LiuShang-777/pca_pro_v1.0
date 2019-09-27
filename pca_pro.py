# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:02:14 2019

@author: liushang
"""
print('>===the PCA analysis is initiated===<')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
print('***packages have been imported correctly***')
parser=argparse.ArgumentParser(description=
                               'the pca analysis which is not depended on sklearn')
parser.add_argument('-df','--dataframe_file',type=str,help='the path of the dataframe')
parser.add_argument('-r','--result',type=str,help='the path of the result')
parser.add_argument('-cl','--classify',type=str,help='the name of the classification column')
parser.add_argument('-s','--start',type=int,help='the start index of columns in transformed matrix')
parser.add_argument('-e','--end',type=int,help='the end index of columns in transformed matrix')
parser.add_argument('-n','--number',type=int,help='the number of component remain in pca')
parser.add_argument('-l','--label',help='the label of different classes in classification column',nargs='*',type=int)
parser.add_argument('-c','--color',help='the color of the responded labels in classification column',nargs='*')
parser.add_argument('-o','--out',help='the output of files in PCA analysis process',choices=['split_file','preprocess_file',
                                                                                            'cov_mat','trans_mat','data_pca'],nargs='*')
args=parser.parse_args()
dataframe_file=args.dataframe_file
result=args.result
start=args.start
end=args.end
n=args.number
label=args.label
color=args.color
classify=args.classify
#文件分割
if args.classify!=None:
    def split_dataframe(dataframe_file,colname):
        classify=[]
        dataframe=pd.read_csv(dataframe_file,sep='\t')
        col_list=dataframe.columns.tolist()
        col_list.remove(colname)
        for i in dataframe[colname]:
            classify.append(i)
        dataframe=dataframe[col_list]
        return dataframe,classify
    dataframe,classify=split_dataframe(args.dataframe_file,args.classify)
    #文件分割产出结果
    if 'split_file' in args.out:        
        def split_dataframe_out(dataframe,classify,result):
            dataframe.to_csv(result+'/dataframe_split.csv',index=False)
            with open(result+'/classify_split.txt','w') as file:
                for i in classify:
                    file.writelines(str(i)+'\n')
        split_dataframe_out(dataframe,classify,result)
else:
    dataframe=pd.read_csv(dataframe_file,sep='\t')
#拿到分类文件转换成分类表
def get_classify(classify_file):
    classify_list=[]
    with open(classify_file,'r') as file:
        for line in file:
            line=line.strip()
            classify_list.append(line)
    return classify_list
#只有X的值预处理
def preprocess(dataframe,start,end):
    x=dataframe.iloc[:,start:end].values
    for i in range(x.shape[1]):
        x[:,i]=(x[:,i]-x[:,i].mean())/np.sqrt(x[:,i].std())
    return x
preprocessed_dataframe=preprocess(dataframe,start,end)
if 'preprocess_file' in args.out:   
    #输出预处理结果
    def preprocess_out(matrix,result):
        dataframe=pd.DataFrame(matrix)
        dataframe.to_csv(result+'/dataframe_preprocess.csv',index=False)
    preprocess_out(preprocessed_dataframe,result)
#拿到协方差矩阵
def covariance_matrix(std_mat):
    mean=np.mean(std_mat,axis=0)
    cov=(std_mat-mean).T.dot((std_mat-mean))/(std_mat.shape[0]-1)
    return cov   
covx=covariance_matrix(preprocessed_dataframe)
#输出协方差矩阵
if 'cov_mat' in args.out: 
    def covariance_matrix_out(cov,result):
        cov_dataframe=pd.DataFrame(cov)
        cov_dataframe.to_csv(result+'/cov_out.csv',index=False)
    covariance_matrix_out(covx,result)    
#标准化流程,
#拿到特征值与特征向量对应的词典
def eig_get(cov_mat):    
    eig_vals,eig_vecs=np.linalg.eig(cov_mat)
    dic={}
    for i in range(len(eig_vals)):
        dic[eig_vals[i]]=eig_vecs[:,i]
    return dic
test_dic=eig_get(covx)
#对特征向量进行排序
def sort_key(test_dic):
    sort_list=[]    
    for i in test_dic.keys():
        sort_list.append(i)
    sort_list.sort(reverse=True)
    return sort_list
sort_list=sort_key(test_dic)
#计算出各主成分考虑进去后的方差解释率
def calculate_explain(sort_list):
    sum_list=0
    for i in sort_list:
        sum_list+=i
    total=sum_list
    explain=[]
    for i in sort_list:
        explain.append(round((i*100/total),2))
    return explain
explain_list=calculate_explain(sort_list)
#解释方差后对结果可视化
def visual_explain(explain_list,n,result):
    plt.figure(figsize=(15,10))
    plt.bar(range(n),explain_list[:n],color='b',label='individual feature for variance explaination')
    plt.step(range(n),np.cumsum(explain_list[:n]),label='cumulative variance was explained')
    plt.ylabel('Ratio of explained variance')
    plt.xlabel('components remained')
    plt.legend(loc='best')
    plt.savefig(result+'/variance_bar.png')
    plt.clf()
visual_explain(explain_list,n,result)
#得到转换矩阵
def trans_mat(sort_list,n):
    concat_list=[]
    for i in range(n):
        concat_list.append(test_dic[sort_list[i]].reshape(len(sort_list),1))
    concat_list=tuple(concat_list)
    matrix_tran=np.hstack(concat_list)
    return matrix_tran
matrix_tran=trans_mat(sort_list,n)
if 'trans_mat' in args.out:   
    #可以尝试输出转化矩阵
    def trans_mat_out(matrix_tran,result):
        matrix_tran_dataframe=pd.DataFrame(matrix_tran)
        matrix_tran_dataframe.to_csv(result+'/tran_dataframe.csv',index=False)
    trans_mat_out(matrix_tran,result)
#用转换矩阵转换原始数据
def data_pca(data,matrix_tran):
    return data.dot(matrix_tran)
pca_result=data_pca(preprocessed_dataframe,matrix_tran)
if 'data_pca' in args.out:   
    def data_pca_out(result_frame,result):
        result_dataframe=pd.DataFrame(result_frame)
        result_dataframe.to_csv(result+'/result_dataframe.csv',index=False)
    data_pca_out(pca_result,result)
if args.classify !=None:
    #拼接数据表
    def to_csv(matrix,classify,result):
        matrix_dataframe=pd.DataFrame(matrix)
        matrix_dataframe['pca_class']=classify    
        matrix_dataframe.to_csv(result+'/pca_classify_result.csv',index=False)
    to_csv(pca_result,classify,result)
else:
    print('there\'s no classify column,so we won\'t perform plot')
#3d绘图
if (classify!=None)&(n==3):   
    def plot_3d(dataframe_file,label,color,result):   
        from mpl_toolkits.mplot3d import Axes3D
        print('>===packages for 3d plot has been imported===<')
        dataframe=pd.read_csv(dataframe_file,sep=',')
        dataframe['pca_class']=classify
        ax=plt.subplot(projection='3d')
        for lab,col in zip(label,color):
            ax.scatter(dataframe[dataframe['pca_class']==lab].iloc[:,0],
                        dataframe[dataframe['pca_class']==lab].iloc[:,1],
                        dataframe[dataframe['pca_class']==lab].iloc[:,2],
                        label=lab,
                        color=col)
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.set_zlabel('PCA3')
        plt.savefig(result+'/pca_3d.png')
    plot_3d(result+'/pca_classify_result.csv',label,color,result)
if (classify!=None)&(n==2):
#2d绘图
    def plot_2d(dataframe_file,label,color,result):
        dataframe=pd.read_csv(dataframe_file,sep=',')
        plt.figure(figsize=(15,10))
        for lab,col in zip(label,color):
            plt.scatter(dataframe[dataframe['pca_class']==lab].iloc[:,0],
                        dataframe[dataframe['pca_class']==lab].iloc[:,1],
                        label=lab,
                        color=col)
            plt.xlabel('PCA1')
            plt.ylabel('PCA2')
            plt.savefig(result+'/pca_2d.png')
    plot_2d(result+'/pca_classify_result.csv',label,color,result)  
print('>==the PCA analysis has been finished==<')
    

