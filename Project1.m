%size of each picture 
m = 250;
n = 250;



avg = zeros(m*n,1);


%%Load Apples 
count=0;
greenAppleLocation = 'C:\Users\nicej\Documents\School\Y4S2\172\Project1\archive\fruits-360-original-size\fruits-360-original-size\Training\apple_6';
greenApples = imageDatastore(greenAppleLocation);
greenApples = augmentedImageDatastore([250 250], greenApples, "ColorPreprocessing","rgb2gray");
data= readall(greenApples);
averageGreenApple = zeros(250*250,1);
greenA = [];
N = 315;
for j=1:N
    r = data(j,1);
    
    r = r{:,:};
    r = cast(cell2mat(r),'double');
    rt = reshape(r,250*250,1);
    greenA = [greenA,rt];
    avg = avg+rt;
    count = count+1;
end

redDeliciosLocation = 'C:\Users\nicej\Documents\School\Y4S2\172\Project1\archive\fruits-360-original-size\fruits-360-original-size\Training\apple_red_delicios_1';
redDelicios = imageDatastore(redDeliciosLocation);
redDelicios = augmentedImageDatastore([250 250], redDelicios, "ColorPreprocessing","rgb2gray");
data= readall(redDelicios);
averageRedDelicious = zeros(250*250,1);
redA = [];
N = 300;
for j=1:N
    r = data(j,1);
    disp(j);
    r = r{:,:};
    r = cast(cell2mat(r),'double');
    rt = reshape(r,250*250,1);
    redA = [redA,rt];
    avg = avg+rt;
    count = count+1;
end

%Load cucumber

cucumberLocation = 'C:\Users\nicej\Documents\School\Y4S2\172\Project1\archive\fruits-360-original-size\fruits-360-original-size\Training\cucumber_1';
cucumber = imageDatastore(cucumberLocation);
cucumber = augmentedImageDatastore([250 250], cucumber, "ColorPreprocessing","rgb2gray");
data= readall(cucumber);
averagecucumber = zeros(250*250,1);
cucumberA = [];
N = 100;
for j=1:N
    r = data(j,1);
    disp(j);
    r = r{:,:};
    r = cast(cell2mat(r),'double');
    rt = reshape(r,250*250,1);
    cucumberA = [cucumberA,rt];
    avg = avg+rt;
    count = count+1;
end


avg = avg/count;

%%Calculate the Averaged face
avgTs = uint8(reshape(avg,m,n));
colormap('gray');
figure(1),imshow(avgTs,'InitialMagnification',400);

%%Center the sample pics at thje origin
for j = 1:18
    B(:,j) = A(:,j) - avg;
    R = reshape(A(:,j),250,250);
end   

%%complete SVD analysis
[U,S,V] = svd(B,'econ');
lambda = U(:,1:18);
lambda(:,1) = -1*lambda(:,1);
figure(2)
count = 1;
for i=1:4
    for j= 1:4
        subplot(4,4,count)
        imshow(200-uint8(62500*reshape(lambda(:,count),250,250)));
        count = count+1;
    end
end
axis off
imagesc(lambda),colormap('gray');
