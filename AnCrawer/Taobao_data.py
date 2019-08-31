import sys
#sys.setdefaultencoding('utf-8')
import requests
import re 
import json
url='https://rate.tmall.com/list_detail_rate.htm?itemId=570431827889&spuId=974109452&sellerId=3943518676&order=3&currentPage=2&append=0&content=1&tagId=&posi=&picture=&groupId=&ua=098%23E1hvfpvbvnQvUpCkvvvvvjiPRFdy1jrbnLcptjrCPmPWAjrEPFSZgjEmPFLZAjlER2yCvvpvvvvv9phvHnQGqj8uzYswznAs7MNTMP2wBluC2QhvCvvvMMGCvpvVvvpvvhCvmphvLv1GEQvjPwex6aZtn0vHfw1ldEZ78BLZQnFpHF%2BSBiVvQRA1%2B2n79n2IAfUTnZJt9ExreEQanuAQiNoOecX20b2XSfpAOH2%2BFOcn%2B3mtvpvIvvvvvhCvVvvvvUHdphvUoQvvvQCvpvACvvv2vhCv2RvvvvWvphvWg8yCvv9vvUv0ygzmFUyCvvOUvvVva6htvpvhvvvvv8wCvvpvvUmm3QhvCvvhvvv%3D&needFold=0&_ksTS=1566910099674_2147&callback=jsonp2148'
content=requests.get(url).content
cont=requests.get(url).content
rex=re.compile(r'\w+[(]{1}(.*)[)]{1}')
content=rex.findall(cont)[0]
con=json.loads(content,"gbk")
count=len(con['rateDetail']['rateList'])
for i in xrange(count):
    print(con['rateDetail']['rateList'][i]['appendComment']['content'])