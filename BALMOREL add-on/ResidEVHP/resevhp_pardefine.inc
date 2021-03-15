PARAMETER DE_RESEVHP(YYY,SSS,TTT,RRR) 'Residential demand including electric vehicles and heat pumps'%semislash%
$if     EXIST '../data/DE_RESEVHP.inc' $INCLUDE         '../data/DE_RESEVHP.inc';
$if not EXIST '../data/DE_RESEVHP.inc' $INCLUDE '../../base/data/DE_RESEVHP.inc';
%semislash%