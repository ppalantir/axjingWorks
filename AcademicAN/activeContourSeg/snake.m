%读入图像

I = imread('./TwoStage/test.jpg');
Igs = im2double(I);
figure,imshow(Igs)

%手动获取snake轮廓点
x=[];y=[];c=1;N=100;
while c<N
    [xi,yi,button] = ginput(1); %%精确获取轮廓点
    x = [x, xi];  %将获取的点存入x,y集合
    y = [y, yi];
    hold on;
    plot(xi,yi,'ro');
    if(button == 3), %当点击鼠标右键时，取点停止
        break; 
    end
    c = c+1;
end


%将第一个点复制到最后，构成完整的轮廓结构
xy = [x;y];
c = c+1;
xy(:,c) = xy(:,1);
%对轮廓线进行插值
t = 1:c;
ts = 1:0.1:c;
xys = spline(t,xy,ts);
xs = xys(1,:); %初始取点横坐标
ys= xys(2,:); %初始取点纵坐标
%查看插值效果
hold on
temp = plot(x(1),y(1),'ro',xs,ys,'b.');
legend(temp, '原点', '插值点');


%%snake算法主体部分
%图像力——线函数
Eline = Igs; %原图像
%图像力——边函数
[gx, gy] =gradient(Igs);
Eedge = -1* sqrt((gx.*gx+gy.*gy)); %梯度图像
%图像力——终点函数
m1 = [-1,1];
m2 = [-1;1];
m3 = [-1,-2,1];
m4 = [-1;-2;1];
m5 = [1,-1;-1,1];
cx = conv2(Igs, m1, 'same');
cy = conv2(Igs, m2, 'same');
cxx = conv2(Igs, m3, 'same');
cyy = conv2(Igs, m4, 'same');
cxy = conv2(Igs, m5, 'same');
[row, col] = size(Igs);
for i = 1:row
    for j = 1:col
        Eterm(i,j) =(cyy(i,j)*cx(i,j)*cx(i,j) + cxx(i,j)*cy(i,j)*cy(i,j) -2*cxy(i,j)*cx(i,j)*cy(i,j))/(1+cx(i,j)*cx(i,j)+cy(i,j)*cy(i,j)^1.5);
    end
end


wl=0; we=0.4; wt=0;
%计算外部力
Eext = wl*Eline + we*Eedge + wt*Eterm;
%计算梯度
[fx, fy] = gradient(Eext);


%计算五对角状矩阵
xs = xs'; %初始取点横坐标集合转换为列向量
ys = ys';
[m,n] = size(xs);
[mm,nn] = size(fx);


alpha=0.2; beta=0.2; gama=1; kappa=0.1;
b(1)=beta;
b(2)=-(alpha + 4*beta);
b(3)=(2*alpha + 6*beta);%%b(i) 表示v(i)系数，从(i-2)到(i+2)
b(4)=b(2);
b(5)=b(1);


A = b(1)*circshift(eye(m),2);
A = A + b(2)*circshift(eye(m),1);
A = A + b(3)*circshift(eye(m),0);
A = A + b(4)*circshift(eye(m),-1);
A = A + b(5)*circshift(eye(m),-2);


%计算矩阵的逆
[L U] = lu(A+gama.*eye(m));
Ainv = inv(U) * inv(L);


% 画图部分
NIter = 1000;
figure
for i = 1:NIter;
    ssx = gama*xs - kappa*interp2(fx,xs,ys);
    ssy = gama*ys - kappa*interp2(fy,xs,ys);
    %计算新的轮廓点位置
    xs = Ainv*ssx;
    ys = Ainv*ssy;
    
    imshow(I)
    hold on;
    plot([xs; xs(1)],[ys; ys(1)], 'r-');
    hold off;
    pause(0.001)
end
