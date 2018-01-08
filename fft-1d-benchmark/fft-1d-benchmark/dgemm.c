#include "fft.h"
#include <stdio.h>
#include <stdlib.h>

/*******************************************
Program to find the multiplication of two matrices
*******************************************/

#include<stdio.h>

int dgemm()
{
    int a[1][32] , b[32][64] , c[1][64] ;
    int i, j, m=1, n=32, p=32, q=64, k ;
    printf("Enter the no. of rows &columns of first matrix: ");
    scanf("%d%d" , &m , &n );
    printf("Enter the no. of rows & columns of 2nd matrix : ");
    scanf("%d%d" , &p , &q );

    if(n!=p)
    printf("Sorry , multiplication wont be possible...!!!\n") ;

    else
   {
    // input
    for( i=0 ; i<m ; i++)
    {
        for( j=0 ; j<n ; j++ )
        scanf( "%d" , &a[i][j] ) ;
    }

    // 0-1 layer weights
    for( i=0 ; i<p; i++)
    {
        for( j=0 ; j<q ; j++ )
        scanf( "%d" , &b[i][j] ) ;
    }

    // calculate sum
     for( i=0 ; i<m ; i++)
    {
        for( j=0 ; j<n ; j++ )
        {
            c[i][j] = 0;
            for( k=0 ; k<n ; k++ )
            c[i][j] += a[i][k] * b[k][j] ;
        }

    }

    printf("The required matrix is \n") ;

     for( i=0 ; i<m ; i++)
    {
        for( j=0 ; j<n ; j++ )
        printf("%2d  " , c[i][j]) ;
        printf("\n");
    }
   }
    return 0 ;
}
