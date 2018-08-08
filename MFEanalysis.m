%%% This matlab script is used for compare temporal response pattern
%%% between population firing rate and NMDA-type synaptic input
mE2E = zeros(2,2,20010);
mE2I = mE2E; mI2E = mE2E; mI2I = mE2E;

mE2Ett = zeros(2,20010); 
mE2Itt = mE2Ett; mI2Ett = mE2Ett; mI2Itt = mE2Ett;
for it = 1:1:2
    
    for is = 1:1:2
        mE2E(it,is,:) = mEbin_ra(:,is)*DEE(it,is) + mEY(it)*fE(it);
        mE2Ett(it,:)  = squeeze(mE2Ett(it,:)) + reshape(mE2E(it,is,:),1,20010);
        mE2I(it,is,:) = mEbin_ra(:,is)*DIE(it,is) + mIY(it)*fI(it);
        mE2Itt(it,:)  = squeeze(mE2Itt(it,:)) + reshape(mE2I(it,is,:),1,20010);
        
        mI2E(it,is,:) = mIbin_ra(:,is)*DEI(it,is);
        mI2Ett(it,:)  = squeeze(mI2Ett(it,:)) + reshape(mI2E(it,is,:),1,20010);
        mI2I(it,is,:) = mIbin_ra(:,is)*DII(it,is);
        mI2Itt(it,:)  = squeeze(mI2Itt(it,:)) + reshape(mI2I(it,is,:),1,20010);
    end
end

ibin = 2;
figure(1)
subplot(3,1,1);
plot(mEbin_ra(1+(ibin-1)*5000:5000*ibin,1),'b');
ylim([0 40]);
subplot(3,1,2);
plot(NMDAEbin_ra(1+(ibin-1)*5000:5000*ibin,1),'b');
subplot(3,1,3);
plot(VEavgbin_ra(1+(ibin-1)*5000:5000*ibin,1),'b');
% plot(mE2Ett(1,1+(ibin-1)*5000:5000*ibin),'b');
% hold on;
% plot(mI2Ett(1,1+(ibin-1)*5000:5000*ibin),'.-C');

figure(2)
subplot(3,1,1);
plot(mIbin_ra(1+(ibin-1)*5000:5000*ibin,1),'R');
ylim([0 40]);
subplot(3,1,2);
plot(NMDAIbin_ra(1+(ibin-1)*5000:5000*ibin,1),'R');
subplot(3,1,3);
plot(VEavgbin_ra(1+(ibin-1)*5000:5000*ibin,2),'b');
% plot(mE2Itt(1,1+(ibin-1)*5000:5000*ibin),'R');
% hold on;
% plot(mI2Itt(1,1+(ibin-1)*5000:5000*ibin),'.-M');

figure(3)
subplot(3,1,1);
plot(mEbin_ra(1+(ibin-1)*5000:5000*ibin,2),'b');
ylim([0 40]);
subplot(3,1,2);
plot(NMDAEbin_ra(1+(ibin-1)*5000:5000*ibin,2),'b');
subplot(3,1,3);
plot(VEavgbin_ra(1+(ibin-1)*5000:5000*ibin,2),'b');
% plot(mE2Ett(2,1+(ibin-1)*5000:5000*ibin),'b');
% hold on;
% plot(mI2Ett(2,1+(ibin-1)*5000:5000*ibin),'.-C');

figure(4)
subplot(3,1,1);
plot(mIbin_ra(1+(ibin-1)*5000:5000*ibin,2),'R');
ylim([0 40]);
subplot(3,1,2);
plot(NMDAIbin_ra(1+(ibin-1)*5000:5000*ibin,2),'R');
subplot(3,1,3);
plot(mE2Itt(2,1+(ibin-1)*5000:5000*ibin),'R');
hold on;
plot(mI2Itt(2,1+(ibin-1)*5000:5000*ibin),'.-M');