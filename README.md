# 毕业设计
毕业设计-汉语多音字注音研究
## 调研阶段

查了一些国内外的论文，总结了一下其中用到的方法

见《问题认识》.md

## 数据处理

用到的数据见Code里的data目录。

`pinyin.txt` ：汉字拼音库，汉字数：41450， 多音字个数：8570，来自[pinyin-data](https://github.com/mozillazg/pinyin-data)

`polyphones.txt`：将汉字拼音库中的多音字全部提取出来，存入该文件中

`polyphones.json`：将所有多音字和读音存储成json文件

`198801.txt`：人民日报1988年一月的新闻语料，一共有19374条新闻，包含9416个多音字，其中有95种不同的多音字，来自[pkuopendata](http://opendata.pku.edu.cn/dataset.xhtml?persistentId=doi:10.18170/DVN/SEYRX5)

`198801output.txt`：存储每个多音字出现的次数

`news.txt`：将所有包含多音字的新闻全部提取出来，存入该文件