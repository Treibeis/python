#include <stdio.h>

#define MAX 80

void main()
{
    int m, n, i, j, t0=0,t1=0,t2=0;
    char end, PATTERN='*';
    while(t0==0){
        if(t1==0){
            printf("Please input the number of lines, n:\n");
            scanf("%d",&n);
            fflush(stdin);
            if(n<1){
                printf("Input error: n should be no less than one.\n");
                continue;
            }
        }
        t1=1;
        if(t2==0){
            printf("Please input the number of spacings, m:\n");
            scanf("%d",&m);
            fflush(stdin);
            if(m<0){
                printf("Input error: m should be no less than zero.\n");
                continue;
            }
        }
        t2=1;
        if(3*n+m-1>MAX){
            printf("3n+m-1 should not be larger than %d.\n",MAX);
            t1=0;t2=0;
        }
        else{
            t0=1;
        }
    }
    for(i=0;i<n;i++){
        for(j=0;j<n-i-1;j++){
            printf(" ");
        }
        for(j=0;j<2*i+1;j++){
            printf("%c",PATTERN);
        }
        for(j=0;j<m;j++){
            printf(" ");
        }
        for(j=0;j<2*n-2*i-1;j++){
            printf("%c",PATTERN);
        }
        for(j=0;j<i;j++){
            printf(" ");
        }
        printf("\n");
    }
    //fflush(stdin); 清理缓存区，有些编译器不支持此功能，这时可以替换为下面的一行代码。
    printf("Press \"Enter\" to exit.\n");
    while((end=getchar())!='\n'&&end!=EOF);
    //scanf("%c",&end);
}
