#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define Data 380
#define TestData 126
#define In 13
#define Out 1
#define Neuron 80
#define TrainC 40000
#define WAlta 0.8
#define VAlta 1.6
double d_in[Data][In];
double d_out[Data][Out];
double t_in[TestData][In];
double t_out[TestData][Out];
double pre[TestData][Out];
double v[Neuron][In];
double y[Neuron];
double w[Out][Neuron]; 
double Maxin[In], Minin[In];
double Maxout[In], Minout[In];
double OutputData[Out];
double dw[Out][Neuron], dv[Neuron][In];
double mse, rmse;

void ReadData()
{
    FILE *fp1, *fp2;
    int i,j; 
    char ch;
    if ((fp1 = fopen("in.txt", "r")) == NULL)
    {
        printf("in.txt open error");
        exit(-1);
    }
    for (i = 0; i < Data; i++)
    {
        for (j = 0; j < In; j++)
        {
            if (j != 0)
                fscanf(fp1, "%c", &ch); //逗号
            fscanf(fp1, "%lf", &d_in[i][j]);
        }
    }
    fclose(fp1);
    if ((fp2 = fopen("out.txt", "r")) == NULL)
    {
        printf("out.txt open error");
        exit(-1);
    }
    for (i = 0; i < Data; i++)
    {
        for (j = 0; j < Out; j++)
            fscanf(fp2, "%lf", &d_out[i][j]);
    }
    fclose(fp2);
}

void InitBPNetwork()
{
	int i,j;
    srand((int)time(NULL));
    for (i = 0; i < In; i++)
    {
        Minin[i] = Maxin[i] = d_in[0][i];
        for (j = 0; j < Data; j++)
        {
            Maxin[i] = Maxin[i] > d_in[j][i] ? Maxin[i] : d_in[j][i];
            Minin[i] = Minin[i] <= d_in[j][i] ? Minin[i] : d_in[j][i];
        }
    }
    for (i = 0; i < Out; i++)
    {
        Minout[i] = Maxout[i] = d_out[0][i];
        for (j = 0; j < Data; j++)
        {
            Maxout[i] = Maxout[i] > d_out[j][i] ? Maxin[i] : d_out[j][i];
            Minout[i] = Minout[i] <= d_out[j][i] ? Minin[i] : d_out[j][i];
        }
    }
    for (i = 0; i < In; i++)
        for (j = 0; j < Data; j++)
            d_in[j][i] = (d_in[j][i] - Minin[i]) / (Maxin[i] - Minin[i]);
    for (i = 0; i < Out; i++)
        for (j = 0; j < Data; j++)
            d_out[j][i] = (d_out[j][i] - Minout[i]) / (Maxout[i] - Minout[i]);
    for (i = 0; i < Neuron; i++)
    {
        for (j = 0; j < In; j++)
        {
            v[i][j] = rand() * 2.0 / RAND_MAX - 1;
            dv[i][j] = 0;
        }
    }
    for (i = 0; i < Out; i++)
    {
        for (j = 0; j < Neuron; j++)
        {
            w[i][j] = rand() * 2.0 / RAND_MAX - 1;
            dw[i][j] = 0;
        }
    }
}

void TrainNetwork()
{
    void ComputO(int var);
    void BackUpdate(int var);
    int count = 1,i,j;
    do
    {
        mse = 0;
        for (i = 0; i < Data; i++)
        {
            ComputO(i);
            BackUpdate(i);
            for (j = 0; j < Out; j++)
            {
                double tmp1 = OutputData[j] * (Maxout[0] - Minout[0]) + Minout[0];
                double tmp2 = d_out[i][j] * (Maxout[0] - Minout[0]) + Minout[0];
                mse += (tmp1 - tmp2) * (tmp1 - tmp2);
            }
        }
        mse = mse / Data * Out;
        if (count % 10000 == 0)
            printf("%d  %lf\n", count, mse);
        count++;
    } while (count!=TrainC);
    printf("training ended,training acount is %d\n",count);
}

void ComputO(int var)
{
    double sum;
    int i,j;
    for (i = 0; i < Neuron; i++)
    {
        sum = 0;
        for (j = 0; j < In; j++)
            sum += v[i][j] * d_in[var][j];
        y[i] = 1 / (1 + exp(-1 * sum));
    }
    for (i = 0; i < Out; i++)
    {
        sum = 0;
        for (j = 0; j < Neuron; j++)
            sum += w[i][j] * y[j];
        OutputData[i] = 1 / (1 + exp(-1 * sum));
    }
}

void BackUpdate(int var)
{
    double t;
    int i,j;
    for (i = 0; i < Neuron; i++)
    {
        t = 0;
        for (j = 0; j < Out; j++)
        {
            dw[j][i] = WAlta * (d_out[var][j] - OutputData[j]) * OutputData[j] * (1 - OutputData[j]) * y[i];
            t += (d_out[var][j] - OutputData[j]) * OutputData[j] * (1 - OutputData[j]) * w[j][i];
        }
        for (j = 0; j < In; j++)
        {
            dv[i][j] = VAlta * t * y[i] * (1 - y[i]) * d_in[var][j];
            v[i][j] += dv[i][j];
        }
    }
}

void TestNetwork()
{
	int i,j,k;
    FILE *fp;
    char ch;
    if ((fp =fopen("test.txt", "r"))  == NULL)
    {
        printf("test.txt open error");
        exit(-1);
    }
    for (i = 0; i < TestData; i++)
    {
        for (j = 0; j < In + Out; j++)
        {
            if (j != 0)
                fscanf(fp, "%c", &ch);
            if (j < In)
                fscanf(fp, "%lf", &t_in[i][j]);
            else
                fscanf(fp, "%lf", &t_out[i][j-In]);
        }
    }
    fclose(fp);
    double sum;
    for (i = 0; i < In; i++)
        for (j = 0; j < TestData; j++)
            t_in[j][i] = (t_in[j][i] - Minin[i]) / (Maxin[i] - Minin[i]);
    for (k = 0; k < TestData; k++)
    {
        for (i = 0; i < Neuron; i++)
        {
            sum = 0;
            for (j = 0; j < In; j++)
                sum += v[i][j] * t_in[k][j];
            y[i] = 1 / (1 + exp(-1 * sum));
        }

        sum = 0;
        for (j = 0; j < Neuron; j++)
            sum += w[0][j] * y[j];
        pre[k][0] = 1 / (1 + exp(-1 * sum)) * (Maxout[0] - Minout[0]) + Minout[0];
        printf("ID:%d predictive value:%.2lf actual value:%.2lf\n", k + 1, pre[k][0], t_out[k][0]);
    }

    rmse = 0.0;
    for (k = 0; k < TestData; k++)
        rmse += (pre[k][0] - t_out[k][0]) * (pre[k][0] - t_out[k][0]);
    rmse = sqrt(rmse / TestData);
    printf("rmse: %.4lf\n", rmse);
}


int main()
{
    ReadData();
    InitBPNetwork();
    TrainNetwork();
    TestNetwork();
    int tmprmse;
    return 0;
}
