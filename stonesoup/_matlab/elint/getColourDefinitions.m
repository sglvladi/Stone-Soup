function colours = getColourDefinitions()

mins2sec = 60;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

colours(1).name = 'freq3';
colours(1).range = [3 3.1];
colours(1).isSwitch = false;
colours(1).isHarmonic = false;

colours(2).name = 'freq9';
colours(2).range = [9 9.1];
colours(2).isSwitch = false;
colours(2).isHarmonic = false;

colours(3).name = 'scanperiod3';
colours(3).range = [0.1 0.5];
colours(3).isSwitch = false;
colours(3).isHarmonic = true;
colours(3).harmonicLogProbs = log([0.9 0.09 0.01]);

colours(4).name = 'scanperiod9';
colours(4).range = [0.1 0.5];
colours(4).isSwitch = false;
colours(4).isHarmonic = true;
colours(4).harmonicLogProbs = log([0.9 0.09 0.01]);

colours(5).name = 'pri3';
colours(5).range = [1e-3 5e-3];
colours(5).isSwitch = true;
colours(5).priorSwitchProb = 0.1;
colours(5).switchRate0to1 = 1/days2sec;
colours(5).switchRate1to0 = 10/days2sec;
colours(5).isHarmonic = false;

colours(6).name = 'pri9';
colours(6).range = [1e-3 5e-3];
colours(6).isSwitch = true;
colours(6).priorSwitchProb = 0.1;
colours(6).switchRate0to1 = 1/days2sec;
colours(6).switchRate1to0 = 10/days2sec;
colours(6).isHarmonic = false;

colours(7).name = 'pulsewidth3';
colours(7).range = [1e-6 5e-6];
colours(7).isSwitch = true;
colours(7).priorSwitchProb = 0.1;
colours(7).switchRate0to1 = 1/days2sec;
colours(7).switchRate1to0 = 10/days2sec;
colours(7).isHarmonic = false;

colours(8).name = 'pulsewidth9';
colours(8).range = [1e-6 5e-6];
colours(8).isSwitch = true;
colours(8).priorSwitchProb = 0.1;
colours(8).switchRate0to1 = 1/days2sec;
colours(8).switchRate1to0 = 10/days2sec;
colours(8).isHarmonic = false;

for i = 1:numel(colours)
    colours(i).q = 0;
    %colours(i).measCov = (0.1*diff(colours(i).range)).^2;
    colours(i).measCov = (0.01*diff(colours(i).range)).^2;
    [colours(i).mean, colours(i).cov] = fitNormalToUniform(....
        colours(i).range(1), colours(i).range(2));
end
