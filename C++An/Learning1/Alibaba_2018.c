/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-08-30 18:50:12
 * @LastEditTime: 2019-08-30 18:50:12
 * @LastEditors: your name
 */
#include <iostream>
#include <set>
#include <string>
#include <vector>
using namespace std;

void mincut(const string& str, const set<string>& dict)
{
    if (str.empty() || dict.empty()) {
        cout << "n/a";
        return;
    }
    int left = 0,right = 1, len = str.size();
    vector<string> selections;
    while(right < len){
        string word = str.substr(left,right-left);
        if(dict.find(word)!=dict.end()){
            int e = right + 1;
            while(e <= len && dict.find(str.substr(left,e-left))==dict.end())
                e++;
            if(e <= len) {
                selections.push_back(str.substr(left,e-left));
                left = right = e;
            }
            else{
                selections.push_back(word);
                left = right;
            }
        }
        right++;
    }
    cout<<right<<" "<<left<<endl;
    if(right-left>1){
        cout<<"n/a";
    }
    else{
        len = selections.size();
        for(int i=0; i<len; i++)
            cout<<selections[i]<<" ";
    }
}

int main(int argc, const char * argv[])
{
    string strS;
    string dictStr;
    int nDict;
    set<string> dict;
    cin >> strS;
    cin >> nDict;
    for (int i = 0; i < nDict; i++)
    {
        cin >> dictStr;
        dict.insert(dictStr);
    }
    mincut(strS, dict);
    return 0;
}



"""
大致意思是给定一组树节点，形式为（id, pId, cost），其中id是节点的序号，pId是父节点的序号，cost是花费，即完成这件事情所需的花费。给定的节点可以构成一棵树或者一个森林，需要求出树或森林中最大的花费是多少。举例如下：
给定节点：
1, 0, 2
2, 1, 3
3, 1, 2
4, 0, 3
构成如下森林：

                (1, 0, 2)               (4, 0, 3)
               _____|_____
              |           |
         (2, 1, 3)     (3, 1, 2)

耗费最大的肯定在叶节点，id=2的节点总花费为父节点2+本身花费3=5，以此类推，id=3的节点花费为4，id=4的节点花费为3，所以给定这组节点花费的最大花费是5。

（我描述的可能不是很准确，大致意思吧）"""

public class AliTest {

    public static void main(String[] args) {
        // 待输入list
        ArrayList<Integer> idList = new ArrayList<>();
        ArrayList<Integer> pIdList = new ArrayList<>();
        ArrayList<Integer> costList = new ArrayList<>();

        // 手动构造测试数据
        idList.add(1);
        idList.add(2);
        idList.add(3);
        idList.add(4);

        pIdList.add(0);
        pIdList.add(1);
        pIdList.add(1);
        pIdList.add(0);

        costList.add(2);
        costList.add(3);
        costList.add(2);
        costList.add(3);

        int get = resolve(idList, pIdList, costList);
        System.out.println(get);
    }

    public static int resolve(ArrayList<Integer> ids, ArrayList<Integer> parents, ArrayList<Integer> costs) {
        Map<Integer, Node> nodeMap = new HashMap<>();
        for(int i = 0; i < ids.size(); i++) {
            Node node = new Node();
            node.setId(ids.get(i));
            node.setParentId(parents.get(i));
            node.setCost(costs.get(i));
            node.setIsleaf(1);
            if(nodeMap.containsKey(parents.get(i))) {
                Node pNode = nodeMap.get(parents.get(i));
                pNode.setIsleaf(0);
                nodeMap.put(parents.get(i), pNode);
            }
            nodeMap.put(ids.get(i), node);
        }

        List<Integer> costList = nodeMap.values()
                .stream()
                .filter(node -> node.getIsleaf() == 1)
                .map(node -> {
                    int myCost = node.getCost();

                    int pId = node.getParentId();
                    while(pId != 0) {
                        Node pNode = nodeMap.get(pId);
                        myCost += pNode.getCost();
                        pId = pNode.getParentId();
                    }
                    return myCost;
                }).collect(Collectors.toList());

        return costList.stream().max((h1, h2) -> h1.compareTo(h2)).get();

    }

    // 树节点
    static class Node {
        private int id; //id
        private int parentId; // 父id
        private int cost; // 花费
        private int isleaf; // 是否为叶节点，1表示是叶节点，0表示不是叶节点

        public int getId() {
            return id;
        }

        public void setId(int id) {
            this.id = id;
        }

        public int getParentId() {
            return parentId;
        }

        public void setParentId(int parentId) {
            this.parentId = parentId;
        }

        public int getCost() {
            return cost;
        }

        public void setCost(int cost) {
            this.cost = cost;
        }

        public int getIsleaf() {
            return isleaf;
        }

        public void setIsleaf(int isleaf) {
            this.isleaf = isleaf;
        }
    }

}
