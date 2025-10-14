%% start, setup the direction contains data and tool box

clear all; clc; close all;
MyToolboxDir = fullfile('D:','fNIRS_MDD','bbci_public-master');
DataDir = fullfile('D:','fNIRS_MDD','MDD');
Dir = fullfile('D:','fNIRS_MDD_DL','ExplainableDL')
cd(MyToolboxDir);
startup_bbci_toolbox('DataDir',DataDir,'TmpDir','/tmp/');
SaveDir = fullfile('D:','fNIRS_MDD_DL','ExplainableDL');
cd(Dir);
% cd ..
BTB_memo= BTB;
%% 
idx = [1,2,3,17,18,19,20,33,34,35,4,5,6,7,8,21,22,23,24,36,37,38,39,40,9,10,11,12,13,25,26,27,28,41,42,43,44,45,14,15,16,29,30,31,32,46,47,48];
up_ch = [6,7,8,21,22,23,24,36,37,38,11,12,13,25,26,27,28,41,42,43];
down_ch = [49,50,51,55,56,57,58,63,64,65,52,53,54,59,60,61,62,66,67,68];

opts=detectImportOptions('channel.xlsx');
opts.SelectedVariableNames = {'channel','x','y'};
opts.Sheet='Sheet1';
channel_info=readtable('channel.xlsx',opts);

sidx=string(idx);
ch_x=channel_info.x;
ch_x=ch_x*3;
ch_y=channel_info.y;
ch_y=ch_y*3;

linear_grid_x=linspace(-3*6.5,3*6.5,1300);
linear_grid_y=linspace(-3*1.5,3*1.5,300);

[interp_x,interp_y]=meshgrid(linear_grid_x,linear_grid_y);

method=1; %1: proposed XAI/2: XAI-HBO/3: XAI-HBR

if method==1
    load('Results_CAD_vft.mat')
    relevance=squeeze(mean(relevance,5))*2;
elseif method==2
    load('BestResultsOxy_CAD_vft.mat')
elseif method==3
    load('BestResultsDeoxy_CAD_vft.mat')
end

real_data = zeros(4,48,246,10);

for fold = 1:10
    idx_real=[1,35,2,34,3,33,17,20,18,19,22,23,56,57,21,24,55,58,8,36,51,63,7,37,50,64,6,38,49,65,5,39,4,40,9,45,10,44,11,43,52,68,12,42,53,67,13,41,54,66,25,28,59,62,26,27,60,61,30,31,29,32,16,46,15,47,14,48];

    if fold == 9 || fold == 10
        hc_class = pred_classF(fold,1:6,1);
        mdd_class = pred_classF(fold,7:10,2);

        acc_hc_correct = find(hc_class==1);
        acc_hc_wrong = find(hc_class==0);

        acc_mdd_correct = find(mdd_class==1)+6;
        acc_mdd_wrong = find(mdd_class==0)+6;
    else
        hc_class = pred_classF(fold,1:7,1);
        mdd_class = pred_classF(fold,8:end,2);

        acc_hc_correct = find(hc_class==1);
        acc_hc_wrong = find(hc_class==0);

        acc_mdd_correct = find(mdd_class==1)+7;
        acc_mdd_wrong = find(mdd_class==0)+7;
    end
    
    correct_hc = squeeze(mean(relevance(fold, acc_hc_correct,:,:),2));
    correct_mdd = squeeze(mean(relevance(fold, acc_mdd_correct,:,:),2));
    wrong_hc = squeeze(mean(relevance(fold, acc_hc_wrong,:,:),2));
    wrong_mdd = squeeze(mean(relevance(fold, acc_mdd_wrong,:,:),2));
    
    relevance_TN = squeeze(relevance(fold, acc_hc_correct,:,:));
    relevance_TP = squeeze(relevance(fold, acc_mdd_correct,:,:));
    
    if ndims(relevance_TN)==2
        relevance_TN=reshape(relevance_TN,[1,size(relevance_TN,1),size(relevance_TN,2)]);
    end
    if ndims(relevance_TP)==2
        relevance_TP=reshape(relevance_TP,[1,size(relevance_TP,1),size(relevance_TP,2)]);
    end

    ymax_weight = max(relevance,[],'all')*0.0005;
    ymin_weight = min(relevance,[],'all')*0.0005;
    for m = 1:20
        mean_channel(m,1) = find(idx_real==up_ch(m));
        mean_channel(m,2) = find(idx_real==down_ch(m));

        correct_hc(mean_channel(m,1),:) = (correct_hc(mean_channel(m,1),:)+correct_hc(mean_channel(m,2),:))/2;
        correct_mdd(mean_channel(m,1),:) = (correct_mdd(mean_channel(m,1),:)+correct_mdd(mean_channel(m,2),:))/2;
        wrong_hc(mean_channel(m,1),:) = (wrong_hc(mean_channel(m,1),:)+wrong_hc(mean_channel(m,2),:))/2;
        wrong_mdd(mean_channel(m,1),:) = (wrong_mdd(mean_channel(m,1),:)+wrong_mdd(mean_channel(m,2),:))/2;
        
        relevance_TN(:,mean_channel(m,1),:) = (relevance_TN(:,mean_channel(m,1),:)+relevance_TN(:,mean_channel(m,2),:))/2; 
        relevance_TP(:,mean_channel(m,1),:) = (relevance_TP(:,mean_channel(m,1),:)+relevance_TP(:,mean_channel(m,2),:))/2;
        
    end

    correct_hc(mean_channel(:,2),:)=[];
    correct_mdd(mean_channel(:,2),:)=[];
    wrong_hc(mean_channel(:,2),:)=[];
    wrong_mdd(mean_channel(:,2),:)=[];
    idx_real(mean_channel(:,2))=[];
    relevance_TN(:,mean_channel(:,2),:)=[];
    relevance_TP(:,mean_channel(:,2),:)=[];
    
    RSTN=zeros(size(acc_hc_correct,2),48,246);
    RSTP=zeros(size(acc_mdd_correct,2),48,246);
    for j=1:48
        [M, I] = max(idx(j)==idx_real);
        real_data(1,j,:,fold) = correct_hc(I,:);
        real_data(2,j,:,fold) = correct_mdd(I,:);
        real_data(3,j,:,fold) = wrong_hc(I,:);
        real_data(4,j,:,fold) = wrong_mdd(I,:);
        RSTN(:,j,:)=relevance_TN(:,I,:);
        RSTP(:,j,:)=relevance_TP(:,I,:);
    end
    RS_TN{fold}=RSTN;
    RS_TP{fold}=RSTP;
end

rnan = isnan(real_data);
real_data = squeeze(mean(mean(real_data,4,'omitnan'),3,'omitnan'));
%% TOP 6 relevance scores for TP and TN 
[LRPsort,ind]=sort(real_data,2,'descend');
idx_TN=idx(ind(1,:));
idx_TP=idx(ind(2,:));
relevanceTN=squeeze(mean(cat(1,RS_TN{1:10}),3));
relevanceTP=squeeze(mean(cat(1,RS_TP{1:10}),3));
relevanceTN=relevanceTN(:,ind(1,1:6));
relevanceTP=relevanceTP(:,ind(2,1:6));
meanR_TN=[mean(relevanceTN);idx_TN(1:6)];
stdR_TN=[std(relevanceTN);idx_TN(1:6)];
meanR_TP=[mean(relevanceTP);idx_TP(1:6)];
stdR_TP=[std(relevanceTP);idx_TP(1:6)];


%% plot_averaged
class = {'Correct HC', 'Correct MDD', 'Wrong HC', 'Wrong MDD'};

figure
for ii=1:4
    interp_z=griddata(ch_x',ch_y',squeeze(real_data(ii,:))',interp_x,interp_y,'natural');

    subplot(2,2,ii)
    pcolor(interp_x, interp_y ,interp_z );
    colormap('hot')

    axis([min(ch_x) max(ch_x) min(ch_y) max(ch_y)]);
    caxis([0 0.3*max(real_data,[],'all')])
    shading interp
    axis off
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    set(gca, 'visible', 'off')
    set(gcf,'units','normalized','outerposition',[0 0 1 1]);
    set(gca, 'LooseInset', get(gca, 'TightInset'));
    title(class{ii})
    colorbar
end

figure();
interp_z=griddata(ch_x',ch_y',squeeze(mean(real_data(1:2,:)))',interp_x,interp_y,'natural');
    
pcolor(interp_x, interp_y ,interp_z );
colormap('hot')

axis([min(ch_x) max(ch_x) min(ch_y) max(ch_y)]);
caxis([0 0.3*max(real_data,[],'all')])
shading interp
axis off
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca, 'visible', 'off')
set(gcf,'units','normalized','outerposition',[0 0 1 1]);
set(gca, 'LooseInset', get(gca, 'TightInset'));

colorbar

save('LRP.mat','real_data');

%% plot individuals

% for fold = 1:10
%     idx_real=[1,35,2,34,3,33,17,20,18,19,22,23,56,57,21,24,55,58,8,36,51,63,7,37,50,64,6,38,49,65,5,39,4,40,9,45,10,44,11,43,52,68,12,42,53,67,13,41,54,66,25,28,59,62,26,27,60,61,30,31,29,32,16,46,15,47,14,48];
% 
%     if fold == 9 || fold == 10
%         hc_class = pred_classF(fold,1:6,1);
%         mdd_class = pred_classF(fold,7:10,2);
% 
%         acc_hc_correct = find(hc_class==1);
%         acc_hc_wrong = find(hc_class==0);
% 
%         acc_mdd_correct = find(mdd_class==1)+6;
%         acc_mdd_wrong = find(mdd_class==0)+6;
%     else
%         hc_class = pred_classF(fold,1:7,1);
%         mdd_class = pred_classF(fold,8:end,2);
% 
%         acc_hc_correct = find(hc_class==1);
%         acc_hc_wrong = find(hc_class==0);
% 
%         acc_mdd_correct = find(mdd_class==1)+7;
%         acc_mdd_wrong = find(mdd_class==0)+7;
%     end
%     
%     correct_hc = squeeze(relevance(fold, acc_hc_correct,:,:));
%     correct_mdd = squeeze(relevance(fold, acc_mdd_correct,:,:));
% 
%     ymax_weight = max(relevance,[],'all')*0.0005;
%     ymin_weight = min(relevance,[],'all')*0.0005;
%     for m = 1:20
%         mean_channel(m,1) = find(idx_real==up_ch(m));
%         mean_channel(m,2) = find(idx_real==down_ch(m));
% 
%         correct_hc(:,mean_channel(m,1),:) = (correct_hc(:,mean_channel(m,1),:)+correct_hc(:,mean_channel(m,2),:))/2;
%         correct_mdd(:,mean_channel(m,1),:) = (correct_mdd(:,mean_channel(m,1),:)+correct_mdd(:,mean_channel(m,2),:))/2;
%         
%     end
% 
%     correct_hc(:,mean_channel(:,2),:)=[];
%     correct_mdd(:,mean_channel(:,2),:)=[];
%     idx_real(mean_channel(:,2))=[];
%     
%     lrp_hc=zeros(length(acc_hc_correct),48,246);
%     lrp_mdd=zeros(length(acc_mdd_correct),48,246);
%     
%     for j=1:48
%         [M, I] = max(idx(j)==idx_real);
%         lrp_hc(:,j,:) = correct_hc(:,I,:);
%         lrp_mdd(:,j,:) = correct_mdd(:,I,:);
%     end
%     
%     figure();
%     title('True positive');
%     for i=1:length(acc_mdd_correct)
% 
%         interp_z=griddata(ch_x',ch_y',squeeze(mean(lrp_mdd(i,:,:),3))',interp_x,interp_y,'natural');
%         subplot(3,3,i)
%         pcolor(interp_x, interp_y ,interp_z );
%         colormap('hot')
% 
%         axis([min(ch_x) max(ch_x) min(ch_y) max(ch_y)]);
%         caxis([0 0.5*max(real_data,[],'all')])
%         shading interp
%         axis off
%         set(gca,'xtick',[])
%         set(gca,'ytick',[])
%         set(gca, 'visible', 'off')
%         set(gcf,'units','normalized','outerposition',[0 0 1 1]);
%         set(gca, 'LooseInset', get(gca, 'TightInset'));
%         colorbar
%     end
%     
%     figure();
%     title('True negative');
%     for i=1:length(acc_hc_correct)
% 
%         interp_z=griddata(ch_x',ch_y',squeeze(mean(lrp_hc(i,:,:),3))',interp_x,interp_y,'natural');
%         subplot(3,3,i)
%         pcolor(interp_x, interp_y ,interp_z );
%         colormap('hot')
% 
%         axis([min(ch_x) max(ch_x) min(ch_y) max(ch_y)]);
%         caxis([0 0.5*max(real_data,[],'all')])
%         shading interp
%         axis off
%         set(gca,'xtick',[])
%         set(gca,'ytick',[])
%         set(gca, 'visible', 'off')
%         set(gcf,'units','normalized','outerposition',[0 0 1 1]);
%         set(gca, 'LooseInset', get(gca, 'TightInset'));
%         colorbar
%     end
%    
% end


