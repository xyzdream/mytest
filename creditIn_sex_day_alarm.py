#!/usr/bin/env python
# -*- coding:utf-8 -*-
#【分类变量-性别-所有指标】测试通过
from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext, Row
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types
import time
import calendar
import datetime
import sys
import math
import numpy as np
import pandas as pd
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
Sixzerosix=spark.read.csv("/zyx/Sample_Day_myData",sep=',')
Sixzerosix.createOrReplaceTempView("Sixzerosix")
table="yxt_detail_v31"
Dataset='creditIn'
Variable="性别"
today=datetime.date.today()
oneday=datetime.timedelta(days=31)
oneyear=datetime.timedelta(days=365)
yesterday=today-oneday
day_date=str(yesterday)[0:4]+str(yesterday)[5:7]+str(yesterday)[8:10]
begin=str(yesterday-oneyear)[0:4]+str(yesterday-oneyear)[5:7]+str(yesterday-oneyear)[8:10]
end=str(yesterday-oneday)[0:4]+str(yesterday-oneday)[5:7]+str(yesterday-oneday)[8:10]
#【缺失率】当天---测试成功
day_lackrate=spark.sql("select 1-count(case when _c3!='' then _c3 else null end)/count(1) as nullrate from Sixzerosix where _c21="+str(day_date)+" and _c20='creditIn' and _c0!=''").toPandas().ix[0,0]
#【缺失率】近一年缺失率历史分位数---测试成功
lackrate_arr=spark.sql("select t1._c9,1-t1.nonullnum/t1.totalnum from(select _c9,count(case when _c3!= '' then _c3 else null end) as nonullnum,count(1) as totalnum from Sixzerosix where _c20='creditIn' and _c21 between "+str(begin)+" and "+str(end)+" group by _c9)t1").toPandas()
lackrate_year=np.zeros(8)
lackrate_year[0]=min(lackrate_arr.ix[:,1])
lackrate_year[1]=max(lackrate_arr.ix[:,1])
lackrate_year[2]=np.percentile(lackrate_arr.ix[:,1],1)
lackrate_year[3]=np.percentile(lackrate_arr.ix[:,1],10)
lackrate_year[4]=np.percentile(lackrate_arr.ix[:,1],75)
lackrate_year[5]=np.percentile(lackrate_arr.ix[:,1],95)
lackrate_year[6]=np.percentile(lackrate_arr.ix[:,1],99)
lackrate_year[7]=sum(lackrate_arr.ix[:,1])/len(lackrate_arr.ix[:,1])
#【不规范比例】当天---测试成功
day_informalrate=spark.sql("select (count(case when _c3!='' then _c3 else null end)-count(case when _c3 in ('1','2') then _c3 else null end))/count(1) from Sixzerosix where _c21="+str(day_date)+" and _c20='creditIn' and _c0!=''").toPandas().ix[0,0]
#【不规范比例】历史分位数---测试成功
informalrate_arr=spark.sql("select t1._c9,(t1.nonullnum-t1.normalnum)/t1.totalnum from (select _c9,count(case when _c3!='' then _c3 else null end) as nonullnum,count(case when _c3 in ('1','2') then _c3 else null end) as normalnum,count(1) as totalnum from Sixzerosix where _c21 between "+str(begin)+" and "+str(end)+" and _c20='creditIn' and _c0!='' group by _c9)t1").toPandas()
informalrate_year=np.zeros(8)
informalrate_year[0]=min(informalrate_arr.ix[:,1])
informalrate_year[1]=max(informalrate_arr.ix[:,1])
informalrate_year[2]=np.percentile(informalrate_arr.ix[:,1],1)
informalrate_year[3]=np.percentile(informalrate_arr.ix[:,1],10)
informalrate_year[4]=np.percentile(informalrate_arr.ix[:,1],75)
informalrate_year[5]=np.percentile(informalrate_arr.ix[:,1],95)
informalrate_year[6]=np.percentile(informalrate_arr.ix[:,1],99)
informalrate_year[7]=sum(informalrate_arr.ix[:,1])/len(informalrate_arr.ix[:,1])
#【PSI】当天---测试成功
day_bin2=spark.sql("select round(t1.malenum/t1.totalnum,3),round(t1.femalenum/t1.totalnum,3),round(t1.othersnum/t1.totalnum,3) from(select count(1) as totalnum,count(case when _c3='1' then _c3 else null end) as malenum,count(case when _c3='2' then _c3 else null end) as femalenum,count(case when _c3 not in ('1','2') then _c3 else null end) as othersnum from Sixzerosix where _c21="+str(day_date)+" and _c20='creditIn' and _c0!='' and _c3!='')t1").toPandas()
year_bin2=spark.sql("select round(t1.malenum/t1.totalnum,3),round(t1.femalenum/t1.totalnum,3),round(t1.othersnum/t1.totalnum,3) from(select count(1) as totalnum,count(case when _c3='1' then _c3 else null end) as malenum,count(case when _c3='2' then _c3 else null end) as femalenum,count(case when _c3 not in ('1','2') then _c3 else null end) as othersnum from Sixzerosix where _c21 between "+str(begin)+" and "+str(end)+" and _c20='creditIn' and _c0!='' and _c3!='')t1").toPandas()
day_bin=day_bin2.fillna(0.00001)
year_bin=year_bin2.fillna(0.00001)
psi=0.00
for i in range(0,day_bin.shape[1],1):day_bin.ix[0,i]=day_bin.ix[0,i]+0.00001;year_bin.ix[0,i]=year_bin.ix[0,i]+0.00001;psi=psi+(day_bin.ix[0,i]-year_bin.ix[0,i])*math.log(day_bin.ix[0,i]/year_bin.ix[0,i])
#【缺失率报警等级计算】测试成功
day_lackrate_alarm=[0,0,0,0,0]
day_lackrate_alarm[0]=day_date
day_lackrate_alarm[1]=Dataset
day_lackrate_alarm[2]=Variable
day_lackrate_alarm[3]=day_lackrate
if day_lackrate>2*lackrate_year[6] or day_lackrate>2*lackrate_year[1]:day_lackrate_alarm[4]='9'
elif  day_lackrate>lackrate_year[6] or day_lackrate>lackrate_year[1]:day_lackrate_alarm[4]='7'
elif  day_lackrate>lackrate_year[6]:day_lackrate_alarm[4]='5'
elif  day_lackrate>lackrate_year[5]:day_lackrate_alarm[4]='3'
else:day_lackrate_alarm[4]='0'
#【不规范比例报警等级计算】
day_informalrate_alarm=[0,0,0,0,0]
day_informalrate_alarm[0]=day_date
day_informalrate_alarm[1]=Dataset
day_informalrate_alarm[2]=Variable
day_informalrate_alarm[3]=day_informalrate
if  day_informalrate>2*informalrate_year[6] or day_informalrate>2*informalrate_year[1]:day_informalrate_alarm[4]='9'
elif day_informalrate>informalrate_year[6] or day_informalrate>informalrate_year[1]:day_informalrate_alarm[4]='7'
elif  day_informalrate>informalrate_year[6]:day_informalrate_alarm[4]='5'
elif  day_informalrate>informalrate_year[5]:day_informalrate_alarm[4]='3'
else:day_informalrate_alarm[4]='0'
#【日PSI报警等级计算】测试成功
day_psi_alarm=[0,0,0,0,0]
day_psi_alarm[0]=day_date
day_psi_alarm[1]=Dataset
day_psi_alarm[2]=Variable
day_psi_alarm[3]=psi
if day_lackrate>0.9:day_psi_alarm[4]='0'
elif psi>0.2:day_psi_alarm[4]='9'
elif psi>0.1:day_psi_alarm[4]='7'
elif psi>0.05:day_psi_alarm[4]='5'
elif psi>0.01:day_psi_alarm[4]='3'
else:day_psi_alarm[4]='0'
#【最终报警】测试成功(任意数量报警都可)
alarm_arr=[int(day_lackrate_alarm[4]),int(day_informalrate_alarm[4]),int(day_psi_alarm[4])]
creditIn_sex_day_alarm=[0,0,0,0,0,0,0]
creditIn_sex_day_alarm[0]=day_date
creditIn_sex_day_alarm[1]=Dataset
creditIn_sex_day_alarm[2]=Variable
max_index=[]
max_alarm_level=max(alarm_arr)
if sum(alarm_arr)>0:
    for i in range(0,len(alarm_arr),1):
        if alarm_arr[i]==max(alarm_arr):max_index.append(i);
else:max_index.append(-1)
if sum(max_index)>-1:
    for j in range(0,len(max_index),1):
        if alarm_arr[max_index[j]]==max(alarm_arr) and max_index[j]==0:creditIn_sex_day_alarm[3]='报警最大等级';creditIn_sex_day_alarm[6]=max(alarm_arr);
        elif alarm_arr[max_index[j]]==max(alarm_arr) and max_index[j]==1:creditIn_sex_day_alarm[4]='报警最大等级';creditIn_sex_day_alarm[6]=max(alarm_arr);
        else:creditIn_sex_day_alarm[5]='报警最大等级';creditIn_sex_day_alarm[6]=max(alarm_arr);
else:creditIn_sex_day_alarm[3]='none';creditIn_sex_day_alarm[4]='none';creditIn_sex_day_alarm[5]='none';creditIn_sex_day_alarm[6]='0'
###【明细数据】输出最终结果---测试成功
x=2
data_allindex=[day_date,table,Dataset,Variable,round(day_lackrate,x),round(day_informalrate,x),round(psi,x)]
data_index_lackrate_detail=[day_date,table,Dataset,Variable,'缺失率',day_lackrate_alarm[4],round(day_lackrate,x),round(lackrate_year[0],x),round(lackrate_year[7],x),round(lackrate_year[1],x),round(lackrate_year[2],x),round(lackrate_year[3],x),round(lackrate_year[4],x),round(lackrate_year[5],x),round(lackrate_year[6],x)]
data_index_informalrate_detail=[day_date,table,Dataset,Variable,'不规范率',day_informalrate_alarm[4],round(day_informalrate,x),round(informalrate_year[0],x),round(informalrate_year[7],x),round(informalrate_year[1],x),round(informalrate_year[2],x),round(informalrate_year[3],x),round(informalrate_year[4],x),round(informalrate_year[5],x),round(informalrate_year[6],x)]
data_index_psi_detail=[day_date,table,Dataset,Variable,'PSI',day_psi_alarm[4],round(psi,x)]
creditIn_sex_day_df_allindex=pd.DataFrame(data_allindex).T
creditIn_sex_day_df_index_lackrate_detail=pd.DataFrame(data_index_lackrate_detail).T
creditIn_sex_day_df_index_informalrate_detail=pd.DataFrame(data_index_informalrate_detail).T
creditIn_sex_day_df_index_psi_detail=pd.DataFrame(data_index_psi_detail).T
#输出报警结果到txt中---测试成功
creditIn_sex_alarm_name=["缺失率","不规范率","PSI"]
f1=open("/home/hadoop/zyx/yxt_detail_v31_day_alarm_"+str(day_date)+".txt","a+")
if creditIn_sex_day_alarm[6]>0:
    f1.write(str(day_date)+"\t"+str(Dataset)+"\t"+str(Variable)+"\t"+str(max_alarm_level)+"\t");
    for i in range(0,len(max_index),1):f1.write(str(creditIn_sex_alarm_name[max_index[i]])+"\t")
else:f1.write("");
f1.write("\n")
f1.close()
#输出明细结果1(当前指标值)到txt中---测试成功
f2=open("/home/hadoop/zyx/yxt_detail_v31_allindex_"+str(day_date)+".txt","a+")
for i in range(0,int(creditIn_sex_day_df_allindex.shape[1]),1):creditIn_sex_day_df_allindex.ix[0,i]=str(creditIn_sex_day_df_allindex.ix[0,i]);f2.write(creditIn_sex_day_df_allindex.ix[0,i]+"\t");
f2.write("\n")
f2.close()
#输出明细结果2（历史分位数等）到txt中---测试成功
f=open("/home/hadoop/zyx/yxt_detail_v31_index_detail_"+str(day_date)+".txt","a+")
for i in range(0,int(creditIn_sex_day_df_index_lackrate_detail.shape[1]),1):creditIn_sex_day_df_index_lackrate_detail.ix[0,i]=str(creditIn_sex_day_df_index_lackrate_detail.ix[0,i]);f.write(creditIn_sex_day_df_index_lackrate_detail.ix[0,i]+"\t");
f.write("\n")
for i in range(0,int(creditIn_sex_day_df_index_informalrate_detail.shape[1]),1):creditIn_sex_day_df_index_informalrate_detail.ix[0,i]=str(creditIn_sex_day_df_index_informalrate_detail.ix[0,i]);f.write(creditIn_sex_day_df_index_informalrate_detail.ix[0,i]+"\t");
f.write("\n")
for i in range(0,int(creditIn_sex_day_df_index_psi_detail.shape[1]),1):creditIn_sex_day_df_index_psi_detail.ix[0,i]=str(creditIn_sex_day_df_index_psi_detail.ix[0,i]);f.write(creditIn_sex_day_df_index_psi_detail.ix[0,i]+"\t");
f.write("\n")
f.close()
#输出明细结果3（PSI相关分布）到txt中
f3=open("/home/hadoop/zyx/yxt_detail_v31_PSI_detail_"+str(day_date)+".txt","a+")
f3.write(table+"\t"+Dataset+"\t"+Variable+"\t"+"当前分布"+"\t")
for i in range(0,int(day_bin.shape[1]),1):day_bin.ix[0,i]=str(round(day_bin.ix[0,i],3));f3.write(day_bin.ix[0,i]+"\t");
f3.write("\n")
f3.write(table+"\t"+Dataset+"\t"+Variable+"\t"+"历史分布"+"\t")
for i in range(0,int(year_bin.shape[1]),1):year_bin.ix[0,i]=str(round(year_bin.ix[0,i],3));f3.write(year_bin.ix[0,i]+"\t");
f3.write("\n")
f3.close()
