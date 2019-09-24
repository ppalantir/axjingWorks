import java.util.Arrays;
import java.util.Scanner;

public class _Qualcomm_1 {
    Scanner sc = new Scanner(System.in);
    int N = sc.nextInt();
    int M = sc.nextInt();
    int a[] = new int[N];
    for (int i = 0; i<N; i++) {
        a[i] = sc.nextInt();
    }
    for (int i = 0; i<M; i++) {
        int flag = sc.nextInt();
        int index = sc.nextInt();
        if (flag == 0) {
            Arrays.sort(a, 0, index);
        } else {
            Arrays.sort(a, 0, index);
            change(a, index);
        }
    }
    for (int i = 0; i<N; i++) {
        System.out.print(a[i] + " ");
    }
}

private static viod change(int[] a, int index) {
    for (int j = 0; j<index/2; j++) {
        int tmp = a[j];
        a[j] = a[index - 1 - j];
        a[index - 1- j] = tmp;
    }
    
}