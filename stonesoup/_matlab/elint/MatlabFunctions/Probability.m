% Simon's Probability class

classdef Probability
    properties
        logval
    end
    methods
        function obj = Probability(arg1,arg2,arg3)
            if(nargin==1)
                %make a Probability from a double (for eg p = Probability(0.4);)
                val = arg1;
                if(length(val)>1)
                    error('Probability: Cannot initialise a Probability with more than a single double');
                end
                obj.logval = log(val);
            end
            if(nargin==2)
                %make an empty array of Probability instances (for eg p = Probability(2,3);)
                obj = Probability(arg1,arg2,NaN);
            end
            if(nargin==3)
                %make an empty array of Probability instances populated with a specific value 
                %(for eg p = Probability(2,3,0.2);)
                n = arg1;
                m = arg2;
                %Paul to vectorise
                for i=1:n
                    for j=1:m
                        obj(i,j) = Probability(arg3);
                    end
                end
            end
        end
        function s = sum(vals)%currently assumes vals is a 1xN or Nx1 vector
            s = Probability(0);
            s.logval = vals(1).logval;
            for i=2:length(vals)
                s.logval = rawplus(s.logval,vals(i).logval);
            end
        end
        function c = times(a,b)%c=a.*b
            checksizes(a,b);
            c = Probability(size(a,1),size(a,2));
            %be good if Paul could vectorise
            for i=1:size(a,1)
                for j=1:size(a,2)
                    c(i,j).logval = a(i,j).logval + b(i,j).logval;
                end
            end
        end
        function c = plus(a,b)%c=a+b [currently have no minus operator by design]
            checksizes(a,b);
            c = Probability(size(a,1),size(a,2));
            %be good if Paul could vectorise
            for i=1:size(a,1)
                for j=1:size(a,2)
                    c(i,j).logval = rawplus(a(i,j).logval,b(i,j).logval);
                end
            end
        end
        function c = rdivide(a,b) %c=a./b
            checksizes(a,b);
            c = Probability(size(a,1),size(a,2));
            %be good if Paul could vectorise
            for i=1:size(a,1)
                for j=1:size(a,2)
                    c(i,j).logval = a(i,j).logval-b(i,j).logval;
                end
            end
        end
        function c = power(a,b) %c=a.^b [assume b is scalar]
            if(length(b)>1)
                error('Probability: exponent must be scalar');
            end
            c = Probability(size(a,1),size(a,2));
            %be good if Paul could vectorise
            for i=1:size(a,1)
                for j=1:size(a,2)
                    c(i,j).logval = a(i,j).logval.*b;
                end
            end
        end
        function b = todouble(a)%supports matrices of Probabilities
            %be good if Paul could vectorise
            for i=1:size(a,1)
                for j=1:size(a,2)
                    b(i,j) = exp(a(i,j).logval);
                end
            end
        end
        function isless = lt(a,b) %a<b
            if(length(a)>1 || length(b)>1)
                error('Probability: less than must operate on scalars');
            end
            isless = a.logval<b.logval;
        end
    end
end

function c = rawplus(loga,logb)
    if(loga<logb)
        smaller = loga;
        bigger = logb;
    else
        smaller = logb;
        bigger = loga;
    end
    if(isinf(bigger) && isinf(smaller))%0+0
        c = bigger;
    else
        c = bigger+log(1+exp(smaller-bigger));
    end
end

function checksizes(a,b)
    if(all(size(a)==size(b)))
        return;
    else
        error('Probability: size incomaptibility')
    end
end