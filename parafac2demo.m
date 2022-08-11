function errorMat = parafac2demo(tnsr)

errorMat = zeros(10,7);
sz = size(tnsr);

parfor ii = 1:10 %cross-validation loop
    loO = zeros(1, sz(3));
    expt = randi(sz(3));
    loO(expt) = 1;

    testTnsr = tnsr(:,:,loO==1);
    trainTnsr = tnsr(:,:,loO==0);

    for kk = 1:7 %These are the number of factors
        [~,A,~, Bs, ~, ~, ~] = nnparafac2(trainTnsr,kk);
        [B, D, P] = parafac2predict(testTnsr, A, Bs);
        errorMat(ii, kk) = 1 - ((norm(testTnsr - B * D * A', "fro")^2 + norm(B - P*Bs,"fro")^2) / norm(testTnsr,'fro')^2);
    end
end
end

function [B, D, P] = parafac2predict(X,A,Bs)
%X is the matrix that needs to be predicted. Simple code, just one this
%time.
%A is matrix of mass spectra (J x rnk)
%Bs is the coupling matrix (rnk x rnk)

%What is the rank
K = size(A,2);
I = size(X,1);

eps = 1e-12;
maxiter = 1000;

for ii = 1:10 %Repeat this 10 times; find the best solution.
%Initialize B
B = rand(I, K);
B = B ./ norm(B);

%Calculate P
[U, ~, V] = svds(B * Bs,K);
P = U * V';

%Initialize D
D = eye(K);

%Estimate mk
% Xas = (X - mean(X)) ./ std(X);
% Xas(isnan(Xas)) = 0;
% 
% S = svds(Xas,2);
% SNR = S(1)/S(2);

%mk = 10^(-SNR/10) * norm(X - B * D * A','fro')^2 / norm(B - P * Bs,'fro')^2;
mk = 1; %We're not using the coupling constant here. Too difficult to estimate.

Xnorm = sum(X(:).^2);
ssr1 = sum(sum((X - B * D * A').^2)) + mk*sum(sum((B-P*Bs).^2)); %updated cost function
ssr1 = ssr1/Xnorm;
ssr2 = 1;
iter = 1;

%Diagnostic purposes
SSR = [];

    while abs(ssr1 - ssr2)/ssr1 > eps && abs(ssr1 - ssr2) > eps && iter < maxiter

        ssr1 = ssr2;

        if iter == 1
        else
            [U, ~, V] = svds(B * Bs,K);
            P = U * V';
        end

        %Calculate B
        B = (X * A * D + mk*P*Bs) * pinv(D * (A' * A) * D + mk * eye(K));
        B = B ./ norm(B);
        B(isnan(B)) = 0;

        %Calculate D
        D = pinv(B'*B) * B' * X * A * pinv(A'*A);
        D = diag(diag(D));

        ssr2 = norm(X - B * D * A', 'fro')^2 + mk * norm(B - P * Bs, 'fro')^2;
        ssr2 = ssr2/Xnorm;

        SSR(iter) = ssr2; %#ok

        iter = iter + 1;
    end

    Bbest(:,:,ii) = B; %#ok
    Dbest(:,:,ii) = D; %#ok
    Pbest(:,:,ii) = P; %#ok
    Smin(ii) = min(SSR); %#ok
end

[~, mndx] = min(Smin);

B = Bbest(:,:,mndx);
D = Dbest(:,:,mndx);
P = Pbest(:,:,mndx);

end

