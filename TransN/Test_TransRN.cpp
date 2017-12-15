#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<pthread.h>
using namespace std;
#define THREADS_NUM 16

bool debug=false;
bool L1_flag=1;

string version;
string data_path = "../../FB15k/";
string trainortest = "test";

map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<string,string> mid2name,mid2type;
map<int,map<int,int> > entity2num;
map<int,int> e2num;
map<pair<string,string>,map<string,double> > rel_left,rel_right;
double lsum_thread[THREADS_NUM], rsum_thread[THREADS_NUM];
double lp_n_thread[THREADS_NUM], rp_n_thread[THREADS_NUM];
double lsum_filter_thread[THREADS_NUM], rsum_filter_thread[THREADS_NUM];
double lp_n_filter_thread[THREADS_NUM], rp_n_filter_thread[THREADS_NUM];
int piece_length;
int relation_num,entity_num;
int n= 100;
double con_alpha = 0.01;

double sigmod(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	return sqrt(res);
}

void vec_output(vector<double> a)
{
	for (int i=0; i<a.size(); i++)
	{
		cout<<a[i]<<"\t";
		if (i%10==9)
			cout<<endl;
	}
	cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

double norm(vector<double> &a)
{
	double x = vec_len(a);
	if (x>1)
	for (int ii=0; ii<a.size(); ii++)
		a[ii]/=x;
	return 0;
}
char buf[100000],buf1[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
	return a.second<b.second;
}


map<pair<int,int>, map<int,int> > ok;
vector<int> fb_h,fb_l,fb_r;
vector<vector<double> > relation_vec,entity_vec, relation_graph_vec;
map<int, vector<int> > relation_neighbors;
class Test{
    vector<int> h,l,r;
public:
    void add(int x,int y,int z, bool flag)
    {
    	if (flag)
    	{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
        }
        ok[make_pair(x,z)][y]=1;
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    double len;
    double get_sum(int rel2, int rel1)
    {
	double sum=0;
	if (L1_flag)
		for(int ii=0; ii<n; ii++)
			sum+=fabs(relation_vec[rel2][ii]-relation_vec[rel1][ii]);
	else
		for (int ii=0; ii<n; ii++)
			sum+=sqr(relation_vec[rel2][ii]-relation_vec[rel1][ii]);
	return sum;
    }
    void get_neighbors_attention_vec()
    {
	relation_graph_vec = relation_vec;
	for(int i=0; i<relation_num; i++)
	{
		if(relation_neighbors.count(i) == 0)
			continue;
		vector<double> weight(relation_neighbors[i].size());
		double sum = 0;
		for(int k=0; k<relation_neighbors[i].size(); k++)
		{
			sum += exp(-get_sum(relation_neighbors[i][k],i));
			weight.push_back(exp(-get_sum(relation_neighbors[i][k], i)));
		}
		for(int k=0; k<n; k++)
		{
			relation_graph_vec[i][k] = 0.0;
			for(int j = 0; j<relation_neighbors[i].size(); j++)
				relation_graph_vec[i][k] += weight[i]/sum*relation_vec[relation_neighbors[i][j]][k];
		}
		norm(relation_graph_vec[i]);
	}
    }
    static double calc_sum(int e1,int e2,int rel)
    {
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            		sum+=-fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]*(1-con_alpha)-con_alpha*relation_graph_vec[rel][ii]);
        else
        	for (int ii=0; ii<n; ii++)
            		sum+=-sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]*(1-con_alpha)-con_alpha*relation_graph_vec[rel][ii]);
        return sum;
    }
    static void *split_test(void *tid_void)
    {
        long tid = (long) tid_void;
	int start_index = tid*piece_length;
	int end_index = (tid+1)*piece_length;
	for (int testid = start_index; testid<end_index; testid+=1)
	{
		int h = fb_h[testid];
		int l = fb_l[testid];
		int rel = fb_r[testid];
		vector<pair<int,double> > a;
		for (int i=0; i<entity_num; i++)
		{
			double sum = calc_sum(i,l,rel);
			a.push_back(make_pair(i,sum));
		}
		sort(a.begin(),a.end(),cmp);
		int filter = 0;
		for (int i=a.size()-1; i>=0; i--)
		{
		    if (ok[make_pair(a[i].first,rel)].count(l)==0)
		    	filter+=1;
			if (a[i].first ==h)
			{
				lsum_thread[tid]+=a.size()-i;
				lsum_filter_thread[tid]+=filter+1;
				if (a.size()-i<=10)
					lp_n_thread[tid]+=1;
				if (filter<10)
					lp_n_filter_thread[tid]+=1;
				break;
			}
		}
		a.clear();
		for (int i=0; i<entity_num; i++)
		{
			double sum = calc_sum(h,i,rel);
			a.push_back(make_pair(i,sum));
		}
		sort(a.begin(),a.end(),cmp);
		filter=0;
		for (int i=a.size()-1; i>=0; i--)
		{
		    if (ok[make_pair(h,rel)].count(a[i].first)==0)
		    	filter+=1;
			if (a[i].first==l)
			{
				rsum_thread[tid]+=a.size()-i;
				rsum_filter_thread[tid]+=filter+1;
				if (a.size()-i<=10)
					rp_n_thread[tid]+=1;
				if (filter<10)
					rp_n_filter_thread[tid]+=1;
				break;
			}
		}
        }
    }
    void run()
    {
        FILE* f1 = fopen(("relation2vec."+version).c_str(),"r");
        FILE* f3 = fopen(("entity2vec."+version).c_str(),"r");
        cout<<relation_num<<' '<<entity_num<<endl;
        int relation_num_fb=relation_num;
        relation_vec.resize(relation_num_fb);
	relation_graph_vec.resize(relation_num_fb);
        for (int i=0; i<relation_num_fb;i++)
        {
            relation_vec[i].resize(n);
	    relation_graph_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f1,"%lf",&relation_vec[i][ii]);
        }
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_num;i++)
        {
            entity_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f3,"%lf",&entity_vec[i][ii]);
            if (vec_len(entity_vec[i])-1>1e-3)
            	cout<<"wrong_entity"<<i<<' '<<vec_len(entity_vec[i])<<endl;
        }
        fclose(f1);
        fclose(f3);
	get_neighbors_attention_vec();
	time_t beginTime, endTime;
	beginTime = time(NULL);
	double lsum=0 ,lsum_filter= 0;
	double rsum = 0,rsum_filter=0;
	double lp_n=0,lp_n_filter;
	double rp_n=0,rp_n_filter;
	piece_length = fb_l.size() / THREADS_NUM;
        pthread_t threads[THREADS_NUM];
	for(int k=0; k<THREADS_NUM; k++)
	{
		lsum_thread[k] = 0;
 		lsum_filter_thread[k] = 0;
		rsum_thread[k] = 0;
		rsum_filter_thread[k] = 0;
		lp_n_thread[k] = 0;
		lp_n_filter_thread[k] = 0;
		rp_n_thread[k] = 0;
		rp_n_filter_thread[k] = 0;
	}
	for(int k=0; k<THREADS_NUM; k++)
		pthread_create(&threads[k], NULL, split_test, (void *)k);
	for(int k=0; k<THREADS_NUM; k++)
	        pthread_join(threads[k], NULL);
	for(int k=0; k<THREADS_NUM; k++)
	{
		lsum+=lsum_thread[k];
		lsum_filter+=lsum_filter_thread[k];
		rsum+=rsum_thread[k];
		rsum_filter+=rsum_filter_thread[k];
		lp_n+=lp_n_thread[k];
		lp_n_filter+=lp_n_filter_thread[k];
		rp_n+=rp_n_thread[k];
		rp_n_filter+=rp_n_filter_thread[k];
	}
	endTime = time(NULL);
	cout<<"total time: "<<(double)(endTime-beginTime)<<endl;
	cout<<"right:"<<rsum/fb_r.size()<<'\t'<<rp_n/fb_r.size()<<'\t'<<rsum_filter/fb_r.size()<<'\t'<<rp_n_filter/fb_r.size()<<endl;
	cout<<"left:"<<lsum/fb_l.size()<<'\t'<<lp_n/fb_l.size()<<"\t"<<lsum_filter/fb_l.size()<<'\t'<<lp_n_filter/fb_l.size()<<endl;
	cout<<"avg:"<<(rsum/fb_r.size()+lsum/fb_l.size())/2<<"\t"<<(rp_n/fb_r.size()+lp_n/fb_l.size())/2<<"\t"<<(rsum_filter/fb_r.size()+lsum_filter/fb_l.size())/2<<"\t"<<(rp_n_filter/fb_r.size()+lp_n_filter/fb_l.size())/2<<endl;
    }

};
Test test;

void prepare()
{
    FILE* f1 = fopen((data_path+"entity2id.txt").c_str(),"r");
	FILE* f2 = fopen((data_path+"relation2id.txt").c_str(),"r");
	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		id2entity[x]=st;
		mid2type[st]="None";
		entity_num++;
	}
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		relation_num++;
	}
    FILE* f_kb = fopen((data_path+"test.txt").c_str(),"r");
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        fscanf(f_kb,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
        	cout<<"miss relation:"<<s3<<endl;
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],true);
    }
    fclose(f_kb);
    FILE* f_kb1 = fopen((data_path+"train.txt").c_str(),"r");
	while (fscanf(f_kb1,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb1,"%s",buf);
        string s2=buf;
        fscanf(f_kb1,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }

        entity2num[relation2id[s3]][entity2id[s1]]+=1;
        entity2num[relation2id[s3]][entity2id[s2]]+=1;
        e2num[entity2id[s1]]+=1;
        e2num[entity2id[s2]]+=1;
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb1);
    FILE* f_kb2 = fopen((data_path+"valid.txt").c_str(),"r");
	while (fscanf(f_kb2,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb2,"%s",buf);
        string s2=buf;
        fscanf(f_kb2,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb2);
    FILE* f_rg = fopen((data_path+"relation_graph.txt").c_str(), "r");
    while(fscanf(f_rg,"%s", buf) == 1)
    {
	string s1 = buf;
	fscanf(f_rg, "%s", buf);
	string s2 = buf;
 	if (relation2id.count(s1) == 0){
		cout << "miss relation:"<< s1 << endl;
		relation2id[s1] = relation_num;
		relation_num++;
	}
	if (relation2id.count(s2) == 0){
		cout << "miss relation: " << s2 << endl;
		relation2id[s2] = relation_num;
		relation_num++;
	}
	relation_neighbors[relation2id[s1]].push_back(relation2id[s2]);
	relation_neighbors[relation2id[s2]].push_back(relation2id[s1]);
    }
    fclose(f_rg);
}


int main(int argc,char**argv)
{
    if (argc<2)
        return 0;
    else
    {
        version = argv[1];
        prepare();
        test.run();
    }
}

