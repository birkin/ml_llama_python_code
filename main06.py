# -*- coding: utf-8 -*-

"""
Attempts to summarize potentially larger blocks of text, via:
- breaking them into blocks, 
- summarizing each block,
- summarizing all the summaries.

Note: there's an error in an assumption below. I'd thought the maximum-limit for the model was 2,000-ish characters,
but I think it's actually 2,000-ish tokens.

However, the code below, based on taking blocks of 2,000 characters, does work.
"""

import argparse, copy, json, logging, pprint

## imported packages ------------------------------------------------
import nltk
nltk.download( 'punkt' )
from llama_cpp import Llama
from nltk.tokenize import sent_tokenize


logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s [%(module)s-%(funcName)s()::%(lineno)d] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S' )
log = logging.getLogger( '__name__' )


## 1,005 words ------------------------------------------------------
# TEST_STRING = '''
# # Thank you so much. Tonight, more than 200 years after a former colony won the right to determine its own destiny, the task of perfecting our union moves forward. It moves forward because of you. It moves forward because you reaffirmed the spirit that has triumphed over war and depression, the spirit that has lifted this country from the depths of despair to the great heights of hope, the belief that while each of us will pursue our own individual dreams, we are an American family and we rise or fall together as one nation and as one people. Tonight, in this election, you, the American people, reminded us that while our road has been hard, while our journey has been long, we have picked ourselves up, we have fought our way back, and we know in our hearts that for the United States of America the best is yet to come. I want to thank every American who participated in this election, whether you voted for the very first time or waited in line for a very long time. By the way, we have to fix that. Whether you pounded the pavement or picked up the phone, whether you held an Obama sign or a Romney sign, you made your voice heard and you made a difference. I just spoke with Gov. Romney and I congratulated him and Paul Ryan on a hard-fought campaign. We may have battled fiercely, but it's only because we love this country deeply and we care so strongly about its future. From George to Lenore to their son Mitt, the Romney family has chosen to give back to America through public service and that is the legacy that we honor and applaud tonight. In the weeks ahead, I also look forward to sitting down with Gov. Romney to talk about where we can work together to move this country forward. I want to thank my friend and partner of the last four years, America's happy warrior, the best vice president anybody could ever hope for, Joe Biden. And I wouldn't be the man I am today without the woman who agreed to marry me 20 years ago. Let me say this publicly: Michelle, I have never loved you more. I have never been prouder to watch the rest of America fall in love with you, too, as our nation's first lady. Sasha and Malia, before our very eyes you're growing up to become two strong, smart beautiful young women, just like your mom. And I'm so proud of you guys. But I will say that for now one dog's probably enough. To the best campaign team and volunteers in the history of politics. The best. The best ever. Some of you were new this time around, and some of you have been at my side since the very beginning. But all of you are family. No matter what you do or where you go from here, you will carry the memory of the history we made together and you will have the lifelong appreciation of a grateful president. Thank you for believing all the way, through every hill, through every valley. You lifted me up the whole way and I will always be grateful for everything that you've done and all the incredible work that you put in. I know that political campaigns can sometimes seem small, even silly. And that provides plenty of fodder for the cynics that tell us that politics is nothing more than a contest of egos or the domain of special interests. But if you ever get the chance to talk to folks who turned out at our rallies and crowded along a rope line in a high school gym, or saw folks working late in a campaign office in some tiny county far away from home, you'll discover something else. You'll hear the determination in the voice of a young field organizer who's working his way through college and wants to make sure every child has that same opportunity. You'll hear the pride in the voice of a volunteer who's going door to door because her brother was finally hired when the local auto plant added another shift. You'll hear the deep patriotism in the voice of a military spouse who's working the phones late at night to make sure that no one who fights for this country ever has to fight for a job or a roof over their head when they come home. That's why we do this. That's what politics can be. That's why elections matter. It's not small, it's big. It's important. Democracy in a nation of 300 million can be noisy and messy and complicated. We have our own opinions. Each of us has deeply held beliefs. And when we go through tough times, when we make big decisions as a country, it necessarily stirs passions, stirs up controversy. That won't change after tonight, and it shouldn't. These arguments we have are a mark of our liberty. We can never forget that as we speak people in distant nations are risking their lives right now just for a chance to argue about the issues that matter, the chance to cast their ballots like we did today. But despite all our differences, most of us share certain hopes for America's future. We want our kids to grow up in a country where they have access to the best schools and the best teachers. A country that lives up to its legacy as the global leader in technology and discovery and innovation, with all the good jobs and new businesses that follow. We want our children to live in an America that isn't burdened by debt, that isn't weakened by inequality, that isn't threatened by the destructive power of a warming planet. We want to pass on a country that's safe and respected and admired around the world, a nation that is defended by the strongest military on earth and the best troops this -- this world has ever known.
# '''

## full speech (2,157 words) ----------------------------------------
TEST_STRING = '''
Thank you so much. Tonight, more than 200 years after a former colony won the right to determine its own destiny, the task of perfecting our union moves forward. It moves forward because of you. It moves forward because you reaffirmed the spirit that has triumphed over war and depression, the spirit that has lifted this country from the depths of despair to the great heights of hope, the belief that while each of us will pursue our own individual dreams, we are an American family and we rise or fall together as one nation and as one people. Tonight, in this election, you, the American people, reminded us that while our road has been hard, while our journey has been long, we have picked ourselves up, we have fought our way back, and we know in our hearts that for the United States of America the best is yet to come. I want to thank every American who participated in this election, whether you voted for the very first time or waited in line for a very long time. By the way, we have to fix that. Whether you pounded the pavement or picked up the phone, whether you held an Obama sign or a Romney sign, you made your voice heard and you made a difference. I just spoke with Gov. Romney and I congratulated him and Paul Ryan on a hard-fought campaign. We may have battled fiercely, but it's only because we love this country deeply and we care so strongly about its future. From George to Lenore to their son Mitt, the Romney family has chosen to give back to America through public service and that is the legacy that we honor and applaud tonight. In the weeks ahead, I also look forward to sitting down with Gov. Romney to talk about where we can work together to move this country forward. I want to thank my friend and partner of the last four years, America's happy warrior, the best vice president anybody could ever hope for, Joe Biden. And I wouldn't be the man I am today without the woman who agreed to marry me 20 years ago. Let me say this publicly: Michelle, I have never loved you more. I have never been prouder to watch the rest of America fall in love with you, too, as our nation's first lady. Sasha and Malia, before our very eyes you're growing up to become two strong, smart beautiful young women, just like your mom. And I'm so proud of you guys. But I will say that for now one dog's probably enough. To the best campaign team and volunteers in the history of politics. The best. The best ever. Some of you were new this time around, and some of you have been at my side since the very beginning. But all of you are family. No matter what you do or where you go from here, you will carry the memory of the history we made together and you will have the lifelong appreciation of a grateful president. Thank you for believing all the way, through every hill, through every valley. You lifted me up the whole way and I will always be grateful for everything that you've done and all the incredible work that you put in. I know that political campaigns can sometimes seem small, even silly. And that provides plenty of fodder for the cynics that tell us that politics is nothing more than a contest of egos or the domain of special interests. But if you ever get the chance to talk to folks who turned out at our rallies and crowded along a rope line in a high school gym, or saw folks working late in a campaign office in some tiny county far away from home, you'll discover something else. You'll hear the determination in the voice of a young field organizer who's working his way through college and wants to make sure every child has that same opportunity. You'll hear the pride in the voice of a volunteer who's going door to door because her brother was finally hired when the local auto plant added another shift. You'll hear the deep patriotism in the voice of a military spouse who's working the phones late at night to make sure that no one who fights for this country ever has to fight for a job or a roof over their head when they come home. That's why we do this. That's what politics can be. That's why elections matter. It's not small, it's big. It's important. Democracy in a nation of 300 million can be noisy and messy and complicated. We have our own opinions. Each of us has deeply held beliefs. And when we go through tough times, when we make big decisions as a country, it necessarily stirs passions, stirs up controversy. That won't change after tonight, and it shouldn't. These arguments we have are a mark of our liberty. We can never forget that as we speak people in distant nations are risking their lives right now just for a chance to argue about the issues that matter, the chance to cast their ballots like we did today. But despite all our differences, most of us share certain hopes for America's future. We want our kids to grow up in a country where they have access to the best schools and the best teachers. A country that lives up to its legacy as the global leader in technology and discovery and innovation, with all the good jobs and new businesses that follow. We want our children to live in an America that isn't burdened by debt, that isn't weakened by inequality, that isn't threatened by the destructive power of a warming planet. We want to pass on a country that's safe and respected and admired around the world, a nation that is defended by the strongest military on earth and the best troops this -- this world has ever known. But also a country that moves with confidence beyond this time of war, to shape a peace that is built on the promise of freedom and dignity for every human being. We believe in a generous America, in a compassionate America, in a tolerant America, open to the dreams of an immigrant's daughter who studies in our schools and pledges to our flag. To the young boy on the south side of Chicago who sees a life beyond the nearest street corner. To the furniture worker's child in North Carolina who wants to become a doctor or a scientist, an engineer or an entrepreneur, a diplomat or even a president -- that's the future we hope for. That's the vision we share. That's where we need to go -- forward. That's where we need to go. Now, we will disagree, sometimes fiercely, about how to get there. As it has for more than two centuries, progress will come in fits and starts. It's not always a straight line. It's not always a smooth path. By itself, the recognition that we have common hopes and dreams won't end all the gridlock or solve all our problems or substitute for the painstaking work of building consensus and making the difficult compromises needed to move this country forward. But that common bond is where we must begin. Our economy is recovering. A decade of war is ending. A long campaign is now over. And whether I earned your vote or not, I have listened to you, I have learned from you, and you've made me a better president. And with your stories and your struggles, I return to the White House more determined and more inspired than ever about the work there is to do and the future that lies ahead. Tonight you voted for action, not politics as usual. You elected us to focus on your jobs, not ours. And in the coming weeks and months, I am looking forward to reaching out and working with leaders of both parties to meet the challenges we can only solve together. Reducing our deficit. Reforming our tax code. Fixing our immigration system. Freeing ourselves from foreign oil. We've got more work to do. But that doesn't mean your work is done. The role of citizen in our democracy does not end with your vote. America's never been about what can be done for us. It's about what can be done by us together through the hard and frustrating, but necessary work of self-government. That's the principle we were founded on. This country has more wealth than any nation, but that's not what makes us rich. We have the most powerful military in history, but that's not what makes us strong. Our university, our culture are all the envy of the world, but that's not what keeps the world coming to our shores. What makes America exceptional are the bonds that hold together the most diverse nation on earth. The belief that our destiny is shared; that this country only works when we accept certain obligations to one another and to future generations. The freedom which so many Americans have fought for and died for come with responsibilities as well as rights. And among those are love and charity and duty and patriotism. That's what makes America great. I am hopeful tonight because I've seen the spirit at work in America. I've seen it in the family business whose owners would rather cut their own pay than lay off their neighbors, and in the workers who would rather cut back their hours than see a friend lose a job. I've seen it in the soldiers who reenlist after losing a limb and in those SEALs who charged up the stairs into darkness and danger because they knew there was a buddy behind them watching their back. I've seen it on the shores of New Jersey and New York, where leaders from every party and level of government have swept aside their differences to help a community rebuild from the wreckage of a terrible storm. And I saw just the other day, in Mentor, Ohio, where a father told the story of his 8-year-old daughter, whose long battle with leukemia nearly cost their family everything had it not been for health care reform passing just a few months before the insurance company was about to stop paying for her care. I had an opportunity to not just talk to the father, but meet this incredible daughter of his. And when he spoke to the crowd listening to that father's story, every parent in that room had tears in their eyes, because we knew that little girl could be our own. And I know that every American wants her future to be just as bright. That's who we are. That's the country I'm so proud to lead as your president. And tonight, despite all the hardship we've been through, despite all the frustrations of Washington, I've never been more hopeful about our future. I have never been more hopeful about America. And I ask you to sustain that hope. I'm not talking about blind optimism, the kind of hope that just ignores the enormity of the tasks ahead or the roadblocks that stand in our path. I'm not talking about the wishful idealism that allows us to just sit on the sidelines or shirk from a fight. I have always believed that hope is that stubborn thing inside us that insists, despite all the evidence to the contrary, that something better awaits us so long as we have the courage to keep reaching, to keep working, to keep fighting. America, I believe we can build on the progress we've made and continue to fight for new jobs and new opportunity and new security for the middle class. I believe we can keep the promise of our founders, the idea that if you're willing to work hard, it doesn't matter who you are or where you come from or what you look like or where you love. It doesn't matter whether you're black or white or Hispanic or Asian or Native American or young or old or rich or poor, able, disabled, gay or straight, you can make it here in America if you're willing to try. I believe we can seize this future together because we are not as divided as our politics suggests. We're not as cynical as the pundits believe. We are greater than the sum of our individual ambitions, and we remain more than a collection of red states and blue states. We are and forever will be the United States of America. And together with your help and God's grace we will continue our journey forward and remind the world just why it is that we live in the greatest nation on Earth. Thank you, America. God bless you. God bless these United States.
'''

## american-mercury text (2,493 words) -----------------------------
# TEST_STRING = '''
# --- Extracted text for page: 1 of 1. --- Aricrican Public Welfare Asso- ciation, American Public Works Association, American Society for Public Administration, American Society of Planning Officials, Con- ference of Chief Justices, Federation of Tax Administrators, Interna- tional City Managers’ Association, Interstate Clearing House on Mental Health, Municipal Finance Officers Association, National As- sociation of Assessing Officers, Na- tional Association of Housing and Redevelopment Officials, Na- tional Association of Attorneys General, National Legislative Con- ference, National Association of State Budget Officers, National As- sociation of State Purchasing Ofh- cials, National Institute of Mu- nicipal Clerks, and Public Person- nel Association. The plasm of “1313” swarms with an executive elite, including NML, who sit on each others boards, commissions, committees. and secretariats. Subordinate car- riers, whom “1313's” employment service places into key positions in government, carry “1313” policy into every nook and cranny of! American life and local govern mental affairs. When any “1313” policy is bein; opposed, “1313” calls out its shock troops of support. Take the case o states’ rights in “1313's” Governors Conference where, for two years, : governor from a middle westerr state has been demanding of th Conference a strong resolution ad vocating restoration of the consti- tutional balance of powers, includ- ing the return of. certain taxing powers to the sovereign states. In 1957, the GC resolutions com- mittee would not report the pro- constitutional resolution — unani- mously and the statement was bur- ied at a closed executive session of GC and hidden from public view. In 1958, the governor tricd again, proposing that GC adopt a strong resolution “to return taxing pow- ers from the federal government back to the States.” Such decen- tralization is not in accord with “metro’s” collectivistic tax policy. Needless to say, the pro-constitu- tional resolution was knocked down again in “1313's” guberna- torial locker room. When a “1313” project, such as “metro,” goes into action, the “1313” machine is kept in motion by all of its members pulling to- gether aided by contact groups such as the League of California Cities which is a member of Com- munist-linked CIMC. oO iLLusTRATE how “1313” ma- Leaves the Miami-Dade Coun- ty situation: early in 1954, The Metropolitan Miami Municipal Board, composed of well-meaning citizens, through a University of Miami professor, commissioned the university to serve as the agent in supervising a fact-finding study regarding local government which had been declared unable to meet the challenge of the changing times. The consultant contract was awarded to PAS in Chicago, leg- man of “1313.” Scant months later, the “1313” report was ready, pro- posing a “metro” city-county gov- ernmental merger of Dade Coun- ty with Miami City. Paralleling activity took place in the Florida Legislature. An amend- ment cleared the way for a charter that was under revision. Result was a “metro” charter par- alleling the PAS report which in- creased the governing body of the county from five to nine commis- sioners, five elected at large and representing no specific district. Citi- zen boards were stripped of juris- diction and demoted to advisory status. Certain elective officers were abolished, including the tax asses- sor and tax collector, to be filled by appointees. Voters approved the charter 44,- 175 to 42,448. County law became the supreme law, and the law will be enacted as ordinances of the board of county ‘commissioners. Through this power of ordinance, the board will be able to establish policy and provide standards of service over many matters before untouchable, including an unlim- ited level of services, financed by taxation levied in part by a group of nonrepresentative policy mak- ers, whom the “metro” charter moved another step farther from the electorate. County government can actually take over a municipal- ity and provide such services if a city fails to mect standards pre- scribed by the county. At present, the Miami-Dade pan- acea-to-panic = program) is being contested in the courts, where de- termined citizens are sceking re- straining orders to chain the “met- ro” leviathan that the new “metro” charter has created. Sacramento’s “mctro” merger, also vamped by “1313's” PAS, con- tains a strikingly similar pattern that abolishes clective offices and strips citizens of representauon through the technique of selecting “at large” six of an 1l-member gov- erning body. HE COMPLEX varicty of state con- ~ Becton and city and county charters personify stumbling blocks to “1313's” militant — reformists. “1313” is ruefully admitting that no immediate action can be taken upon the Sacramento proposal be- cause the plan requested cannot be attempted without a change in the California Constitution. “Metro” attempted a crash pro- gram in Los Angeles, tried to by- pass a constitutional provision, and aroused the wrath of the citizenry, who were by no means asleep. Ac- cording to California law, only a Board of Freeholders, duly elected by the people, can legally submit a revised charter to the electorate. The Los Angeles Board of Su- pervisors high-handedly appoint- ed a charter committee, ignoring the law, but public disapproval blasted the renegade charter pre- sentcd by the committee, and op- posed the enlarged “metro” 11- member governing body, power- rule by ordinance, abolishment of elective othces and the constabu- lary. and delegation of administra- ive power to an appointed county manager, Which latter position was so loosely drawn that “1313” could send one of its “metro” trainees to fill it. Americans are especially an- gered by “metro’s” proposal to deny citizens the right to elect their sheriff. Most Americans believe that the sheriff, as chief law en- forcement officer in a county, has a direct responsibility to all the peo- ple of that county. “1313” proposes abolishment. of city police forces and formation of a metropolitan police force, prefer- ably under a political appointce— the “metro” sheriff—with an area- wide centralized communications system under his control. Tradi- tionally, as at present in the U.S.A, police power is delegated by the citizenry to law enforcement ‘ofh- cers at township, hamlet, or city levels, and the police are held an- swerable to the people whom they serve and protect. “Metro” police power is diametrically opposed to the American principle of a decen- tralized police power responsible to the citizenry it serves. While usually refraining from laying a finger on American schools, “metro” encourages ac- celeration of the modern trend toward consolidation of school dis- tricts, a movement completely in accord with “metro” objectives, because “metro’s” ultimate design for schools is complete subjugation under the “metro” consolidated budget. The contemporary one-court pro- posal for the judiciary appears to be another movement paralleling “metro” policy. Vice-chairman of “1313's” Conference of Chief Jus- tices, as recently as August, 195%, asked his fellow chicf justices to join in “modernizing the Ameri- can legal system.” ne prive of “1313” to force Wi chateake in present law continues. State constitution study groups (a current LWV_ project), charter study and revision committees, and press agentry that ballyhoos various types of centralization are forms of “1313” activity that occur when “metro” sets its sieht on a target area. During such campaigns, NML’s “model” laws, state consti- tution, city and county charters are leaned upon heavily, as well as canned editorials distributed free by NML to newspaper editors. Absurdly, “1313” appears to be predicating its pro-appointed man- ager arguments upon the ridicu- lous assumption that most elected executives are stupid or unreliable while: practically all appointed ad- ininistrators are bright and hon- est. NML’s “model” state constitu- uion actually proposes an appornt- cd state manager, presumably to as- sume the responsibility of an inept governor, who, according to the same NML “model”, can delegate all administrative power to the ap- pointee. “Metro” gets a toe in the civic door by persuading a city or county to appoint managers vested with administrative power. Following this, “metro’s” political leukemia can invade the remainder of the governmental system, proposing “metro” features, promoting con- solidations, annexations, and above all—“metro” mergers. This type of action leads to shrinkage of the total of grass-root governmental units throughout the Nation. In 1957, units of government in the United States totaled 102,353, as compared to 155,116 in 1942. The average annual decrease in the last 15 years has been 2.3 per cent, and in the last five, 2.5 per cent. “Met- ro” would hurry the process. scaTrer chart of the 168-plus A metropolitan areas in the United States shows the Los An- geles-San Diego-San Bernardino patch as the geographically largest. Thickest concentration of metro- politan areas occurs on the Atlan- tic-Mississippi sector of the Nation. “Metro” mergers, not already at- tempted, are being scheduled in these and other areas, with “1313's” PAS angling for the consultant contracts, but not getting them in every instance. When such is the case, a local “1313” carrier may step forward as a volunteer consultant, perhaps a radical political scientist, an insti- tute of public administration, or a university bureau of governmen- tal research, any of which may have access to the local bin of source materials. A typical bin is the 16-volume survey of Los Angeles, conducted over a period of years under the guidance of an NML chairman stationed in California, who secured financial backing from a tax-exempt foundation. “Metro” material such as this provides the makings for a quick rehash, by PAS or any other “1313” carrier, “when a “meétro” campaign strikes town. In addition to Miami-Dade and Sacramento, PAS has served “mcet- re” promotions im northcasiern Illinois, Fairfax County, Virginia, and Jefferson Parish in Louisiana. At Nashville and Seattle, where PAS “experts” did not act as con- sultants, “metro” was voted down. Ulustrating further the complex- ity of situations possible under a “metro” invasion, is the troubled situation in Toronto, Canada. Fol- lowing establishinent of “metro,” worried Canadians, like — their American counterparts in Miami- Dade, are trying to regain control of their governmental affairs. “Metro” appeals to the populace by promising thrift, claiming that “metro” big government is frugal government because it abolishes unnecessary units of government. Actually, merging does not dimin- ish the line base of services and their cost—only cost cutting, re- trenchment, and trimming of ex- travagance and waste can do that. Even top “metro” officials admit that “metro” government actually costs more, HRINKING the number of govern- S mental units does not provide a panacea of lower taxes as the ac- tual case of the State of Nebraska shows. Nebraska leads in total number of governmental units (6,059), yet enjoys the lowest per capita state tax burden of all the states save one, and bears no long- term debt. Certain “1313” adjuncts main- tain branch offices, one in The Hague, Netherlands, another in Havana, several .in Washington, D. C.,, where federal legislation comes under the “1313” whip. In Washington, , ‘'1313’s”’ NAHRO, which is promoting “slum clearance” and community demolishment and rebuilding, has counseled with federal agencies and other policy-making bodies on questions of national and local pol- icy from 1937 up to the current year. Also, “1313's” Interstate Clear- ing House on Mental Health pro- vides impetus to the controversial mental health movement through its mechanism of propaganda, na- tionwide “studies,” and regional working arrangements with organ- izations and state agencies. “1313's” PPA (formerly Civil Service Assembly of the U.S. and Canada) conducts an annual in- ternational meeting and four an- nual regional conferences and lends a broad base of support to the professional carecrists in “1313's” mushrooming empire of public ad- ministration. Actually, it appears that after laying to rest the corrupt remains of the Tweed and Gas Rings, NML has raised from:the ashes the “1313” specter, whose “metro” threat is more terrible than anything that has risen in the past. What reasons are given for pro- moung such a form of government? Charges of “horse and buggy” have been leveled at American constitutions and charters which have been accused of inability to meet the changing times. To these claims have been added the twin bogeys of increasing population and _ population-concentration in urban areas. “1313” crowns this with the charge that present exor- bitant costs of running govern- ment ‘will soar even higher if the “metro” panacea is not accepted. Increasing population is no occa- sion for panic, nor is it an admissi- ble excuse for “metro’s” panacea- to-panic program. Needed in “metro” afflicted areas is alerted thinking to recognize “metro” policy or program wher- ever appearing, and citizen action to forestall the big “metro” inva- sion, for “1313” is on the war march for additional scalps to hang alongside Miami-Dade. Prec- edent set by any unit of American government falling before “metro,” will play squarely into the hands of those who would weaken the American Republic. Also needed is an intelligent citi- zen approach to eliminate bureau- cratic malpractices which are bur- dening Americans with unneces- sary taxation, These corrections are to be accomplished, not by chang- ing the form of government, but by changing sick administration within government. “Metro’s” pill of centralization merely makes the sick patient sicker. An answer to “metro” at its first step is a countermovement de- signed to safeguard individual American rights and to prevent the economic smothering of citizens under crushing “metro” taxation and debt. A tightening of citizen control upon public affairs through city-county separation, is one solu- tion that can be tried. City-county separation runs counter to merging and would make “metro’s” other steps difficult, perhaps impossible. From a taxpayer viewpoint, city- county separation may include the advantages of (1) abolishing the practice of county spending in un- incorporated areas part of the taxes raised from the city, (2) reducing duplication of functions performed by the two governments, (3) bring- ing government closer to the gov- erned-by-consent. Separation could cast county gov- ernment into a miniature of the ideal type of federal government— an agency restricted to handling over-all problems for which it is geographically suited, such as flood control, and forbidding it from usurping municipal servicing at the local level. The political situation would remain at home. Similar corrective steps at other governmental levels now being menaced by “metro” action or pol- icy, such as states’ rights at the federal level, would go far in re- storing government to the hands of the governed and dispel the “metro” monolithic encroachment. INALLY, a revitalization of sound economic thinking is im- perative, to redirect into American society the principles of competitive private enterprise as opposed to “metro” planned public enterprise; also a renaissance in sociopolitical thinking which could whittle bu- reaucracy down to zero size. To accomplish these ends, the American electorate must keep its hands free upon the power of the ballot. Americans have toppled in- ept public officials and will do so again, correct past mistakes, and prevent would-be tyrants from seiz- ing political and economic power.
# '''

## manager function -------------------------------------------------
def summarize_text( text_as_json: str ) -> dict:
    """ Manages summarization.
        Called by dundermain. """
    ## validate input -----------------------------------------------
    text_to_summarize: str = validate_input( text_as_json )
    ## load model ---------------------------------------------------
    LLM = load_model()
    ## summarize ----------------------------------------------------

    ## get the word-count for the text_to_summarize -----------------
    word_count = len( text_to_summarize.split() )
    log.debug( f'word_count, ``{word_count}``' )

    ## get the first 1750 words -------------------------------------
    text_to_summarize = ' '.join( text_to_summarize.split()[0:1000] )  # I want to keep the max-tokens under 2,000
    summarization_dict = summarize( text_to_summarize, LLM )

    # if len( text_to_summarize ) > 1000:
    #     summarization_dict = block_and_summarize( text_to_summarize, LLM )
    # else:
    #     summarization_dict = summarize( text_to_summarize, LLM )
    return summarization_dict


## helper functions -------------------------------------------------


def block_and_summarize( text_to_summarize: str, LLM ) -> dict:
    """ Logic (this function is only called if text_to_summarize is greater than 1,000 characters)...
        - take the first 10,000 characters and split them into roughly equal-sized chunks (keeping complete sentences).
        - run the summarization on each chunk, with a "max_tokens" of 200. 
        - assemble all the summaries together into one string of sentences, and run a summary on that string -- using the original "max_tokens" of 100.
    Called by summarize_text()
    """
    log.debug( 'starting block_and_summarize()' )
    ## assemble sentences -------------------------------------------
    sentences = sent_tokenize( text_to_summarize[0:10000] )
    log.debug( f'sentences, ``{sentences}``' )
    log.debug( f'number of sentences, ``{len(sentences)}``' )
    ## assemble chunks ----------------------------------------------
    chunks = []
    chunk = ''
    for sentence in sentences:
        log.debug( f'processing sentence, ``{sentence}``' )
        if len(chunk) + len(sentence) < 2000:
            log.debug( 'adding sentence to chunk' )
            chunk += sentence + ' '
            log.debug( f'chunk with sentence added is now, ``{chunk}``' )
        else:
            log.debug( f'chunk is full (size ``{len(chunk)}``); adding chunk to chunks, and re-initializing chunk with existing sentence' )
            log.debug( f'chunk-check: len(chunks), ``{len(chunks)}``' )
            if len(chunks) <= 4:
                log.debug( 'chunk-check: will append this chunk' )
                chunks.append( chunk )
            else:
                log.debug( 'chunk-check: would not append this sixth chunk' )
        
            log.debug( f'chunks is now, ``{pprint.pformat(chunks)}``' )
            log.debug( f'number of chunks is now, ``{len(chunks)}``' )
            chunk = sentence + ' '
            log.debug( f'new chunk is, ``{chunk}``' )
    log.debug( f"note: at this point we've gone through all our sentences, and have up to five chunks. There may be an additional chunk in process we're ignoring. In this case it is: ``{chunk}``" )
    # if chunk:
    #     log.debug( f'appending chunk (size ``{len(chunk)}``) to chunks' )
    #     chunks.append( chunk )
    log.debug( f'chunks, ``{pprint.pformat(chunks)}``' )
    log.debug( '\n-----\n'.join(f'Element {index}: {element}' for index, element in enumerate(chunks)) )
    log.debug( f'number of chunks, ``{len(chunks)}``' )

    ## summarize each chunk -----------------------------------------
    summaries = []
    for chunk in chunks:
        summarized_chunk = summarize( chunk, LLM, max_tokens_for_summarization=200 )
        summaries.append( summarized_chunk )
    log.debug( f'summaries, ``{pprint.pformat(summaries)}``' )
    # log.debug( '\n-----\n'.join(f'Element {index}: {element}' for index, element in enumerate(summaries)) )
    ## combine summaries & run final summarization ------------------
    combined_summaries = ''
    for summary in summaries:
        combined_summaries = f"{combined_summaries} {summary['choices'][0]['message']['content']}"
    log.debug( f'\n\ncombined_summaries, ``{combined_summaries}``\n\n' )
    combined_summaries = combined_summaries.replace( 'The text describes', '...' )
    combined_summaries = combined_summaries.replace( 'The text', '...' )
    combined_summaries = combined_summaries.replace( 'the text describes', '...' )
    log.debug( f'\n\ncleaned combined_summaries, ``{combined_summaries}``\n\n' )
    combined_summarization_dict: dict = summarize( combined_summaries, LLM, max_tokens_for_summarization=100 )
    return combined_summarization_dict


def summarize( text_to_summarize: str, LLM, max_tokens_for_summarization=100 ) -> dict:
    """ Summarizes text.
        Explanation of model parameters: <https://chat.openai.com/share/e5923e46-4236-4034-9d99-829ab18dbfdc> (chatgpt4, 2023-Nov-25)
        Called by summarize_text() """
    log.debug( 'starting summarize()' )
    log.debug( f'text_to_summarize, \n\n``{text_to_summarize}``\n\n' )

    word_text = '75'
    if max_tokens_for_summarization == 200:  # we want a more thorough summary for chunks that'll be combined and then summarized
        word_text = '150'

    # message = f'Summarize the following text ({word_text}-word-maximum) using a neutral tone, describing main themes and topics. The text: {f"{text_to_summarize} (end-of-text)"}'
    message = f'Give a three-sentence summary of the following text using a neutral tone, describing main themes and topics. The text: {text_to_summarize}'
    # message = f'Summarize, in one short sentence, using a neutral tone -- the following text: {text_to_summarize}'

    log.debug( f'message, ``{message}``' )
    messages = [ {'role': 'user', 'content': message} ]

    output: dict = LLM.create_chat_completion(
        messages, 
        temperature=0.2, 
        top_p=0.95, 
        top_k=40, 
        stream=False, 
        stop=[], 
        max_tokens=max_tokens_for_summarization,
        repeat_penalty=1.1
        )
    log.debug( f'output, ``{pprint.pformat(output)}``' )

    return output


def validate_input( text_as_json: str ) -> str:
    """ Validates input.
        Called by summarize_text() """
    log.debug( 'validating input' )
    text_to_summarize = None
    if not text_as_json:
        log.info( 'no text_as_json found, will use test Obama re-election speech' )
        text_to_summarize = TEST_STRING
    if text_to_summarize == None:
        try:
            text_to_summarize_dct = json.loads( text_as_json )
        except Exception as e:
            raise Exception( 'input does not appear to be valid json' )
        try:
            text_to_summarize = text_to_summarize_dct['text_to_summarize']
        except Exception as e:
            raise Exception( 'input-json missing required "text_to_summarize" key' )
    log.debug( f'text_to_summarize, ``{text_to_summarize}``' )
    return text_to_summarize


def load_model() -> None:
    """ Loads model.
        Called by summarize_text() """
    log.debug( 'loading model' )
    LLM = Llama( 
        model_path='../models/ggml-vicuna-13b-4bit-rev1.bin',
        n_ctx=4000 )
    assert type(LLM) == Llama, type(LLM)
    log.debug( 'model loaded' )
    return LLM


## dundermain -------------------------------------------------------


if __name__ == '__main__':
    ## set up argparser ---------------------------------------------
    parser = argparse.ArgumentParser( description='summarizes text' )
    parser.add_argument('--text_as_json', type=str, help='json defining text_to_summarize dict-key' )
    args = parser.parse_args()
    log.debug( f'args: {args}' )
    ## get json-string ----------------------------------------------
    text_as_json = args.text_as_json
    ## get to work ---------------------------------------------------
    summarize_text ( text_as_json )
    log.debug( 'done' )


# if __name__ == '__main__':
#     ## set up argparser ---------------------------------------------
#     parser = argparse.ArgumentParser( description='summarizes text' )
#     parser.add_argument('--text_as_json', type=str, help='json defining text_to_summarize dict-key' )
#     args = parser.parse_args()
#     log.debug( f'args: {args}' )
#     ## get json-string ----------------------------------------------
#     text_as_json = args.text_as_json
#     ## validate -----------------------------------------------------
#     try:
#         if text_as_json:
#             json.loads( text_as_json )
#         else:
#             log.debug( 'no text_as_json found' )
#     except Exception as e:
#         raise Exception( 'input does not appear to be valid json' )
#     log.debug( f'text_as_json arg: {text_as_json}' )
#     if text_as_json == None:
#         test_dct = { 'text_to_summarize': TEST_STRING }
#         text_as_json = json.dumps( test_dct )
#     log.debug( f'final text_as_json, ``{text_as_json}``' )
#     log.info( 'using obama re-election speech as test' )
#     ## get to work ---------------------------------------------------
#     summarize_text ( text_as_json )
#     log.debug( 'done' )
