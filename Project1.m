clc; clear all; close all;

%size of each picture 
m = 250;
n = 250;

% Number of sample pictures
N = 96;
avg = zeros(m*n,1);

%%Load zucchini
count = 0;
zucchiniLocation = 'E:\Education2.0\Spring22\ECE172\fruits-360-original-size\fruits-360-original-size\Training\zucchini_1';
zucchinis = imageDatastore(zucchiniLocation);
zucchinis = augmentedImageDatastore([250 250], zucchinis, "ColorPreprocessing","rgb2gray");
data = readall(zucchinis);

A = [];
for j=1:N
    r = data(j,1);
    
    r = r{:,:};
    r = cast(cell2mat(r),'double');
    rt = reshape(r,250*250,1);
    %zucchiniA = [zucchiniA,rt];
    A = [A,rt];
    avg = avg+rt;
    count = count+1;
end

%%Red Apples
redAppleLocation = 'E:\Education2.0\Spring22\ECE172\fruits-360-original-size\fruits-360-original-size\Training\apple_crimson_snow_1';
redApple = imageDatastore(redAppleLocation);
redApple = augmentedImageDatastore([250 250], redApple, "ColorPreprocessing","rgb2gray");
data = readall(redApple);

for j=1:N
    r = data(j,1);
    
    r = r{:,:};
    r = cast(cell2mat(r),'double');
    rt = reshape(r,250*250,1);
    A = [A,rt];
    avg = avg+rt;
    count = count+1;
end

%%cucumber
cucumberLocation = 'E:\Education2.0\Spring22\ECE172\fruits-360-original-size\fruits-360-original-size\Training\cucumber_1';
cucumber = imageDatastore(cucumberLocation);
cucumber = augmentedImageDatastore([250 250], cucumber, "ColorPreprocessing","rgb2gray");
data = readall(cucumber);

for j=1:N
    r = data(j,1);
    
    r = r{:,:};
    r = cast(cell2mat(r),'double');
    rt = reshape(r,250*250,1);
    A = [A,rt];
    avg = avg+rt;
    count = count+1;
end

%%Calculate the average fruit
avg = avg/count;
avgTs = uint8(reshape(avg,m,n));
colormap('gray');
figure(1),imshow(avgTs,'InitialMagnification',400);

%%Center the sample picture at the origin
for j = 1:3*N
    B(:,j) = A(:,j) - avg;
    R = reshape(A(:,j),m,n);
end

%%Compute the SVD
[U,S,V] = svd(B,'econ');
Phi = U(:,1:3*N);
Phi(:,1) = -1*Phi(:,1);
figure(2)
count = 1;
for i = 1:4
    for j = 1:4
        subplot(4,4,count)
        imshow(200-uint8(250000*reshape(Phi(:,count),m,n)),'InitialMagnification',400);
        count = count + 1;
    end
end

%%Project each image onto basis
for j = 1:N
    imvec = B(:,j);
    cucu(:,j) = imvec'*Phi(:,1:3);
end

for j = 1:N
    imvec = B(:,N+j);
    ap(:,j) = imvec'*Phi(:,1:3);
end

for j = 1:N
    imvec = B(:,(2*N)+j);
    zu(:,j) = imvec'*Phi(:,1:3);
end

figure(3)

plot3(cucu(1,:),cucu(2,:),cucu(3,:),'r.','MarkerSize',30)
hold on
plot3(ap(1,:),ap(2,:),ap(3,:),'b.','MarkerSize',30)
plot3(zu(1,:),zu(2,:),zu(3,:),'g.','MarkerSize',30)
xlabel('PCA 1')
ylabel('PCA 2')
zlabel('PCA 3')

legend('CUCUMBER','APPLE','ZUCCHINI')

%%Test new image from internet

%apple pic
u = imread("E:\Education2.0\Spring22\ECE172\MiniProject\testApple.JPG");
u = imresize(u,[250 250]);
figure(4)
subplot(1,3,1)
imshow(u);
u = double(rgb2gray(u));
uApple = reshape(u,m*n,1) - avg;
Applepts = uApple'*Phi(:,1:3);

%cucumber pic
v = imread("E:\Education2.0\Spring22\ECE172\MiniProject\testCucumber.JPG");
v = imresize(v,[250 250]);
subplot(1,3,2)
imshow(v);
v = double(rgb2gray(v));
vCucu = reshape(v,m*n,1) - avg;
Cucupts = vCucu'*Phi(:,1:3);

%zucchini pic
w = imread("E:\Education2.0\Spring22\ECE172\MiniProject\testZucchini.JPG");
w = imresize(w, [250 250]);
subplot(1,3,3)
imshow(w);
w = double(rgb2gray(w));
wZu = reshape(w,m*n,1) - avg;
wZupts = wZu'*Phi(:,1:3);

%%plot new pictures to see where they fall
figure(3)
plot3(Applepts(1),Applepts(2),Applepts(3),'.m','MarkerSize',30)
plot3(Cucupts(1),Cucupts(2),Cucupts(3),'k.','MarkerSize',30)
plot3(wZupts(1),wZupts(2),wZupts(3),'.y','MarkerSize',30)

legend('CUCUMBER','APPLE','ZUCCHINI', 'APPLE NEW', 'CUCUMBER NEW', 'ZUCCHINI NEW')

%%Part B SVM 

allPoints = [ap, cucu, zu];
%1 = apples, 2 = cucumbers, 3 = zucchinis
label = [ones(96,1); 2*ones(96,1); 3*ones(96,1)];

%train svm classifier
features = 1:3;
xtrain = [U(1:96,features); U(97:192,features); U(193:288,features)];

SVMModel = fitcecoc(xtrain,label);
test_labels(1) = predict(SVMModel,Applepts);
test_labels(2) = predict(SVMModel,Cucupts);
test_labels(3) = predict(SVMModel,wZupts);

fprintf("For SVM classification 1 = apples, 2 = cucumbers, 3 = zucchinis\n");

fprintf("New apple image was clasified as %d by SVM\n",test_labels(1));
fprintf("New cucumber image was clasified as %d by SVM\n",test_labels(2));
fprintf("New zucchini image was clasified as %d by SVM\n\n",test_labels(3));

%%Part C Decision Trees

TreeModel = fitctree(xtrain,label,crossval="on");
view(TreeModel.Trained{1}, Mode="graph");

%%Part D

neighborAp = knnsearch(xtrain,Applepts);
neighborCu = knnsearch(xtrain,Cucupts);
neighborZu = knnsearch(xtrain,wZupts);

fprintf(label{neighborAp});
