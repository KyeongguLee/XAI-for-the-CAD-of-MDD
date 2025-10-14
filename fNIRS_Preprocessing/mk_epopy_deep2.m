
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


%% Read the raw data From the directory

Task = {'vft'};
ival = [-5 35; 25 65; -5 35;]*1000;
% ival = [0 10]*1000;
base_ival = [ -1 0; 29 30; -1 0] * 1000;

extract_ival = [0 30; 30 60; 0 30]*1000;
Group = {'정상군','환자군'};

notused{1} = [34,36,38,59];
notused{2} = [1,9,34,52,22];
ch_check = zeros(1,204);

idx=[1,35,2,34,3,33,17,20,18,19,22,23,56,57,21,24,55,58,8,36,51,63,7,37,50,64,6,38,49,65,5,39,4,40,9,45,10,44,11,43,52,68,12,42,53,67,13,41,54,66,25,28,59,62,26,27,60,61,30,31,29,32,16,46,15,47,14,48];
%idx=randperm(68);
%idx=1:68;
%idx=[1,2,3,17,18,22,56,21,55,8,51,7,50,6,49,5,4,9,10,11,52,12,53,13,54,25,59,26,60,30,29,16,15,14,35,34,33,20,19,23,57,24,58,36,63,37,64,38,65,39,40,45,44,43,68,42,67,41,66,28,62,27,61,31,32,46,47,48];
%ridx = [1,2,3,4,5,11,12,13,14,15,16,17,18,21,22];
%lidx = [10,9,8,7,6,27,26,25,30,29,28,20,19,24,23];
rng(1024) 
splitMDD=randperm(48)+68;
splitHC=randperm(68);

for i = 1:size(Task,2)
    
    nSub = 0;
    epo.oxy = [];
    epo.deoxy = [];
    
    for ii = 1:size(Group,2)
        
        DataD = [DataDir '\' Group{ii}]
        subname = ls([DataD '\sub*']);
        for iii = 1 :size(subname,1)
            nSub = nSub + 1;
            
            filename = subname(iii,:);
            
            subnum = str2num(filename(end-1:end));
            
            if(sum(notused{ii}==subnum))
                continue;
            end
            load([DataD '\' filename '\cnt_' Task{i}]);
            
            fs = 1/0.12288;
            cnt.fs = fs; 
            
            %[b,a] = butter(3, 0.5/cnt.fs*2); 
            %cnt = proc_filtfilt(cnt,b,a);
                      
            cnt = proc_BeerLambert(cnt,'Epsilon',1/0.4343*[0.754,1.1075; 1.097 0.781],'DPF',[6, 5.2], 'Opdist', 3);
            
            [b,a] = butter(3,[0.01 0.3]/cnt.fs*2);  
            cnt = proc_filtfilt(cnt,b,a);
            
            cnt.oxy = proc_selectChannels(cnt,'not','*deoxy');
            cnt.oxy.clab = strrep(cnt.oxy.clab,'oxy','');
            cnt.deoxy = proc_selectChannels(cnt,'*deoxy');
            cnt.deoxy.clab = strrep(cnt.deoxy.clab,'deoxy','');
            cnt.oxy = proc_selectChannels(cnt.oxy,idx);
            cnt.deoxy = proc_selectChannels(cnt.deoxy,idx);
            
            epot.oxy = proc_segmentation(cnt.oxy,mrk,ival(i,:));
            epot.oxy = proc_baseline(epot.oxy,base_ival(i,:));
            
            epot.deoxy = proc_segmentation(cnt.deoxy,mrk,ival(i,:));
            epot.deoxy = proc_baseline(epot.deoxy,base_ival(i,:));
            
            nTask(nSub) = size(epot.oxy.x,3);
            if(nTask(nSub)~=3)
                nSub
            end 
            epot.oxy.xUnit = 'mS';
            epot.oxy.yUnit = 'mV';
            epot.deoxy.xUnit = 'mS';
            epot.deoxy.yUnit = 'mV';
            epo.oxy = proc_appendEpochs(epo.oxy,epot.oxy);
            epo.deoxy = proc_appendEpochs(epo.deoxy,epot.deoxy);
        end
    end
    epo.oxy.fs = cnt.fs;
    epo.deoxy.fs = cnt.fs;
    epo.oxy = proc_selectIval(epo.oxy,extract_ival(i,:));
    epo.deoxy = proc_selectIval(epo.deoxy,extract_ival(i,:));
    epo.oxy.y = full(ind2vec(epo.oxy.event.desc'));
    epo.deoxy.y = full(ind2vec(epo.deoxy.event.desc'));
    epo.oxy.x = reshape(epo.oxy.x,size(epo.oxy.x,1),size(epo.oxy.x,2),3,size(epo.oxy.x,3)/3);
    epo.deoxy.x = reshape(epo.deoxy.x,size(epo.deoxy.x,1),size(epo.deoxy.x,2),3,size(epo.deoxy.x,3)/3);
    epo.oxy.y = reshape(epo.oxy.y,size(epo.oxy.y,1),3,size(epo.oxy.y,2)/3);
    epo.deoxy.y = reshape(epo.deoxy.y,size(epo.deoxy.y,1),3,size(epo.deoxy.y,2)/3);
    
    
% Folding
    oxy.x = squeeze(mean(epo.oxy.x,3));
    oxy.y = squeeze(epo.oxy.y(:,1,:));
    %oxy.x = normalize(oxy.x,1);
    for cv=1:8
        testN{cv}=[splitHC((cv-1)*7+1:cv*7), splitMDD((cv-1)*5+1:cv*5)];
        a=testN{cv};
        oxy.testX{cv}=oxy.x(:,:,a);
        testY{cv}=oxy.y(:,a);
        id=ones(116,1);
        id(a)=0;
        oxy.trainX{cv}=oxy.x(:,:,logical(id));
        trainY{cv}=oxy.y(:,logical(id));
    end
    med1=cv*7;
    med2=cv*5;
    for cv=9:10
        testN{cv}=[splitHC(med1+(cv-9)*6+1:med1+(cv-8)*6), splitMDD(med2+(cv-9)*4+1:med2+(cv-8)*4)];
        a=testN{cv};
        oxy.testX{cv}=oxy.x(:,:,a);
        testY{cv}=oxy.y(:,a);
        id=ones(116,1);
        id(a)=0;
        oxy.trainX{cv}=oxy.x(:,:,logical(id));
        trainY{cv}=oxy.y(:,logical(id));
    end
    
    deoxy.x = squeeze(mean(epo.deoxy.x,3));
    deoxy.y = squeeze(epo.deoxy.y(:,1,:));
    %deoxy.x = normalize(deoxy.x,1);
    
    for cv=1:8
        a=testN{cv};
        deoxy.testX{cv}=deoxy.x(:,:,a);
        id=ones(116,1);
        id(a)=0;
        deoxy.trainX{cv}=deoxy.x(:,:,logical(id));
    end
   
    for cv=9:10
        a=testN{cv};
        deoxy.testX{cv}=deoxy.x(:,:,a);
        id=ones(116,1);
        id(a)=0;
        deoxy.trainX{cv}=deoxy.x(:,:,logical(id));
    end
    
    epopy.NumID=testN;
    epopy.trainY=trainY;
    epopy.testY=testY;
    
    for cv=1:10
        epopy.trainX{cv}=cat(2,oxy.trainX{cv},deoxy.trainX{cv});
        epopy.testX{cv}=cat(2,oxy.testX{cv},deoxy.testX{cv});
    end
    save(['epopy_' Task{i} '.mat'],'epopy','-v7.3');
    
end
