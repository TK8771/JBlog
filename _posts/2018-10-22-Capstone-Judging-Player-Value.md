---
layout: post
title: "Capstone: Gauging NHL Player Value"
date: 2018-10-20
excerpt: "Capstone: Gauging NHL Player Valu"
tags:
- Capstone
- NHL
- Linear regression
---
## Introduction

Over the course of the last 15 to 20 years, statistics and probability have fundamentally changed the world of professional sports. This statistical 'revolution' is romanticized in the movie "Moneyball," based on Michael Lewis' book of the same name. In the film, set in 2002, the General Manager of the MLB Oakland Athletics, Billy Beane (played by Brad Pitt) teams up with Peter Brand, a fictious Yale economics graduate major that is based on the real life assistant GM at the time, Paul DePodesta (played by Jonah Hill) to radically alter how the Oakland A's assess player value. In order to compete with MLB teams that have much larger player payrolls, Beane and DePodesta had to find players that were undervalued by the standard collective wisdom of baseballs' scouts, managers, and coaches. As such, they had to apply a new method of evaluating player value.  

They defied commonly followed baseball stats, and the intuition of their scouts, and started to use sabermetrics. Roughly defined, sabermetrics is an empirical analysis of in-game activity. Through advanced statistical analysis, certain indicators (on-base percentage, slugging percentage, etc.) were determined to be better predictors of offensive success than the 'standard' stats (batting avg, stolen bases, etc.). Focusing on these stats, Beane and his staff were able to acquire players that were atypical and undervalued. It helped the Oakland A's to set a 20-game win streak, and successfully compete with franchises with significantly larger payrolls (Yankees, Red Sox, etc.) than themselves. Since then, using sabermetrics has become a pillar of valuing players in Major League Baseball.

Beane and DePondesta effectively changed the landscape of evaluating professional baseball players. A very similar change is occurring in the National Hockey League, albeit late to the table in comparison to the other major sports, it has gained a lot of traction in the last few years. Especially during the 2014-15 season, when the NHL partnered with SAP to create an enhanced statistical package that matched up with launch of a new website featuring advanced analytics. This also coincided with several prominent NHL franchises adding data analytics positions to their front office including Kyle Dubas (Toronto Maple Leafs), Tyler Dellow (Edmonton Oilers), and Sunny Mehta (New Jersey Devils). Since then, it has really taken off.

## The Greatest Game on Earth.

Parity is at all time high in the NHL. When the trade deadline rolls around in late February, many teams still fancy themselves in the hunt for the Stanley Cup. To add to that effect, ever since the LA Kings won the Cup in 2012 as an 8 seed (and went 16-4 no less), many teams believe the whole mantra of "you just have to make it to the playoffs, and then anything can happen."

This inevitably leads to a glut of 'buyers' (teams looking to pickup players to increase their chances of winning) at the trade deadline, but not nearly as many 'sellers' (teams that want to get of rid of players for future draft picks or young prospects). This, in conjunction with the salary cap that I'll discuss later, means finding undervalued players is a quintessential way to try to gain an edge come playoff time.

My hope is that I am able to find the best statistics to understand player value vs how much that player makes.

Which leads me to my data science question: Can you build a model to predict NHL player's salaries? What are the best predictors of how much a player will make?

If I can find undervalued players, that would ideally help a general manager make decisions on who to try to acquire.

## NHL Salary Cap

Introduced after the full-season lockout of 04-05, the NHL currently has a salary cap in place. The main purposes of this salary cap is to curtail player salary growth to a reasonably manageable level, but also allow smaller-market teams to compete with larger-market teams. This cap is referred to as a 'hard' salary cap, meaning that each team can only spend up to that cap amount on a team of (at least) 24 players, up to a maximum of 50 players.

As one of my metrics for understanding player value will be value added vs cap hit, I'll be using the NHL salary cap to gauge how much each player is making as a percentage of their teams' total cap.  

As I was only able to find player salary information going back to the 2011-2012 season, and the cap changes a variable amount every year, I'll be manually entering the salary cap for each year. And in cases where I might need it in varying formats, I'll be entering it once as a full year-to-year label (2011-2012), and once as a single year with the starting year representing the whole season (so, 2011-2012 would equal just 2011). There's probably a better way to do it than this, but it's moot as writing these both out doesn't take much time at all.

## Process / Workflow / Data Collection

**Note: After further research, I ended up finding a better, more detailed site that included extra data points.** This site (http://www.corsica.hockey/) allows its user to extract the information into .csv, rendering this loop, and the subsequent data pulled, redundant. However, since I put a great deal of effort into getting this loop to run properly, I'm going to leave it here to show my work. The code was original run, and it worked, but leaving it off of subsequent runs for timeliness.

I have to pull player salary from a different source than player performance data, so there will be two separate data pulls and therefore loops.

These next dozen cells or so are going to be aimed at testing out functionality before creating a loop.

Base URL pulling from: https://www.spotrac.com/nhl/rankings/cap-hit/  
Subsquent URLs look like this: https://www.spotrac.com/nhl/rankings/YEAR/cap-hit/  
Where YEAR = the year the season opened in.

```python
# Scraper loop
# Original URL
url = 'https://www.hockey-reference.com/play-index/ppbp_finder.cgi?c2stat=&c4stat=&c2comp=&order_by_asc=&game_location=&c1comp=&year_min=2008&request=1&franch_id=&birth_country=&match=single&year_max=2018&c3comp=&report=ppbp&season_end=-1&c3stat=&order_by=player&season_start=1&c1val=&c3val=&c2val=&handed=&rookie=N&pos=S&describe_only=&c1stat=&situation_id=ev&c4val=&age_min=0&age_max=99&c4comp=&offset='
df_puck = pd.DataFrame([], columns=['player_name', 'pos', 'team_id', 'season', 'games_played', 'goals', 'assists', 'points', 'corsi_for', 'corsi_against', 'corsi_pct', 'corsi_rel_pct', 'corsi_per_60', 'corsi_rel_per_60', 'fenwick_for', 'fenwick_against', 'fenwick_pct', 'fenwick_rel_pct', 'on_ice_shot_pct', 'on_ice_sv_pct', 'pdo', 'zs_offense_pct', 'zs_defense_pct', 'toi_pbp_avg', 'faceoff_wins', 'faceoff_losses', 'faceoff_percentage', 'hits', 'blocks', 'takeaways', 'giveaways'])

# See logic above for why I chose these numbers
for i in range(0, 9700, 100):

    # Create lists fresh on each loop
    player_name_list = []
    pos_list = []
    team_id_list = []
    season_list = []
    games_played_list = []
    goals_list = []
    assists_list = []
    points_list = []
    corsi_for_list = []
    corsi_against_list = []
    corsi_pct_list = []
    corsi_rel_pct_list = []
    corsi_per_60_list = []
    corsi_rel_per_60_list = []
    fenwick_for_list = []
    fenwick_against_list = []
    fenwick_pct_list = []
    fenwick_rel_pct_list = []
    on_ice_shot_pct_list = []
    on_ice_sv_pct_list = []
    pdo_list = []
    zs_offense_pct_list = []
    zs_defense_pct_list = []
    toi_pbp_avg_list = []
    faceoff_wins_list = []
    faceoff_losses_list = []
    faceoff_percentage_list = []
    hits_list = []
    blocks_list = []
    takeaways_list = []
    giveaways_list = []

    # Iteration to create end of URL
    next_get = str(i)

    # Request get
    res = requests.get(url+next_get)

    # Create into bs4 object
    soup = BeautifulSoup(res.content, 'lxml')

    # Breakdown soup via find_all into its various pieces
    player_name = soup.find_all('td', {'class':'left', 'data-stat':'player'})
    pos = soup.find_all('td', {'class':'center', 'data-stat':'pos'})
    team_id = soup.find_all('td', {'class':'left', 'data-stat':'team_id'})
    season = soup.find_all('td', {'class':'left', 'data-stat':'season'})
    games_played = soup.find_all('td', {'class':'right', 'data-stat':'games_played'})
    goals = soup.find_all('td', {'class':'right', 'data-stat':'goals'})
    assists = soup.find_all('td', {'class':'right', 'data-stat':'assists'})
    points = soup.find_all('td', {'class':'right', 'data-stat':'points'})
    corsi_for = soup.find_all('td', {'class':'right', 'data-stat':'corsi_for'})
    corsi_against = soup.find_all('td', {'class':'right', 'data-stat':'corsi_against'})
    corsi_pct = soup.find_all('td', {'class':'right', 'data-stat':'corsi_pct'})
    corsi_rel_pct = soup.find_all('td', {'class':'right', 'data-stat':'corsi_rel_pct'})
    corsi_per_60 = soup.find_all('td', {'class':'right', 'data-stat':'corsi_per_60'})
    corsi_rel_per_60 = soup.find_all('td', {'class':'right', 'data-stat':'corsi_rel_per_60'})
    fenwick_for = soup.find_all('td', {'class':'right', 'data-stat':'fenwick_for'})
    fenwick_against = soup.find_all('td', {'class':'right', 'data-stat':'fenwick_against'})
    fenwick_pct = soup.find_all('td', {'class':'right', 'data-stat':'fenwick_pct'})
    fenwick_rel_pct = soup.find_all('td', {'class':'right', 'data-stat':'fenwick_rel_pct'})
    on_ice_shot_pct = soup.find_all('td', {'class':'right', 'data-stat':'on_ice_shot_pct'})
    on_ice_sv_pct = soup.find_all('td', {'class':'right', 'data-stat':'on_ice_sv_pct'})
    pdo = soup.find_all('td', {'class':'right', 'data-stat':'pdo'})
    zs_offense_pct = soup.find_all('td', {'class':'right', 'data-stat':'zs_offense_pct'})
    zs_defense_pct = soup.find_all('td', {'class':'right', 'data-stat':'zs_defense_pct'})
    toi_pbp_avg = soup.find_all('td', {'class':'right', 'data-stat':'toi_pbp_avg'})
    faceoff_wins = soup.find_all('td', {'class':'right', 'data-stat':'faceoff_wins'})
    faceoff_losses = soup.find_all('td', {'class':'right', 'data-stat':'faceoff_losses'})
    faceoff_percentage = soup.find_all('td', {'class':'center', 'data-stat':'faceoff_percentage'})
    hits = soup.find_all('td', {'class':'right', 'data-stat':'hits'})
    blocks = soup.find_all('td', {'class':'right', 'data-stat':'blocks'})
    takeaways = soup.find_all('td', {'class':'right', 'data-stat':'takeaways'})
    giveaways = soup.find_all('td', {'class':'right', 'data-stat':'giveaways'})

    # Add the various soup objects into a new dataframe
    for a in range(0, len(player_name), 1):
        if a == 0:
            df_append = pd.DataFrame(
            {'player_name': player_name[a].text,
            'pos': pos[a].text,
            'team_id': team_id[a].text,
            'season': season[a].text,
            'games_played': games_played[a].text,
            'goals': goals[a].text,
            'assists': assists[a].text,
            'points': points[a].text,
            'corsi_for': corsi_for[a].text,
            'corsi_against': corsi_against[a].text,
            'corsi_pct': corsi_pct[a].text,
            'corsi_rel_pct': corsi_rel_pct[a].text,
            'corsi_per_60': corsi_per_60[a].text,
            'corsi_rel_per_60': corsi_rel_per_60[a].text,
            'fenwick_for': fenwick_for[a].text,
            'fenwick_against': fenwick_against[a].text,
            'fenwick_pct': fenwick_pct[a].text,
            'fenwick_rel_pct': fenwick_rel_pct[a].text,
            'on_ice_shot_pct': on_ice_shot_pct[a].text,
            'on_ice_sv_pct': on_ice_sv_pct[a].text,
            'pdo': pdo[a].text,
            'zs_offense_pct': zs_offense_pct[a].text,
            'zs_defense_pct': zs_defense_pct[a].text,
            'toi_pbp_avg': toi_pbp_avg[a].text,
            'faceoff_wins': faceoff_wins[a].text,
            'faceoff_losses': faceoff_losses[a].text,
            'faceoff_percentage': faceoff_percentage[a].text,
            'hits': hits[a].text,
            'blocks': blocks[a].text,
            'takeaways': takeaways[a].text,
            'giveaways': giveaways[a].text}, index=[i])
        else:
            df_append = df_append.append(
            {'player_name': player_name[a].text,
            'pos': pos[a].text,
            'team_id': team_id[a].text,
            'season': season[a].text,
            'games_played': games_played[a].text,
            'goals': goals[a].text,
            'assists': assists[a].text,
            'points': points[a].text,
            'corsi_for': corsi_for[a].text,
            'corsi_against': corsi_against[a].text,
            'corsi_pct': corsi_pct[a].text,
            'corsi_rel_pct': corsi_rel_pct[a].text,
            'corsi_per_60': corsi_per_60[a].text,
            'corsi_rel_per_60': corsi_rel_per_60[a].text,
            'fenwick_for': fenwick_for[a].text,
            'fenwick_against': fenwick_against[a].text,
            'fenwick_pct': fenwick_pct[a].text,
            'fenwick_rel_pct': fenwick_rel_pct[a].text,
            'on_ice_shot_pct': on_ice_shot_pct[a].text,
            'on_ice_sv_pct': on_ice_sv_pct[a].text,
            'pdo': pdo[a].text,
            'zs_offense_pct': zs_offense_pct[a].text,
            'zs_defense_pct': zs_defense_pct[a].text,
            'toi_pbp_avg': toi_pbp_avg[a].text,
            'faceoff_wins': faceoff_wins[a].text,
            'faceoff_losses': faceoff_losses[a].text,
            'faceoff_percentage': faceoff_percentage[a].text,
            'hits': hits[a].text,
            'blocks': blocks[a].text,
            'takeaways': takeaways[a].text,
            'giveaways': giveaways[a].text}, ignore_index = True)

    # Kept getting timeout errors, so added a sleep to offset
    time.sleep(3)

    df_puck = df_puck.append(df_append, ignore_index = True)
    df_puck.to_csv('hockey_data.csv', index = True)
```
- After further testing, it doesn't seem that above URL can pull more than 100 entries at a time.
- Furthermore, the page uses 'never-ending' scrolling, so the URL never changes once you get past the first 100 entries.
- I'm going to have to create a loop to loop through each year, and I'm going to do it by team name.
- Max amount of contracts an NHL team can have is 50, so this should hopefully work.
- As a result, I'll have to create a list with each team name in it.

## Player Cap Hits

An important distinction to make before diving in here: a player's salary can, and usually is, different then their actual cap hit. A player's cap hit is the average annual value over the entire length of their contract.  

So for example, in 2007, Pittsburgh Penguins' captain Sidney Crosby signed a12-year, 104.4 million dollar contract. The average annual value comes out to8.7 million/year, which is how much his salary counts against the cap.However, the deal is not evenly structured throughout the contract to pay Sidthe Kid 8.7 mil/yr. He was paid 12 mil/yr the first 3 years of the contract,but will only be paid 3 mil/yr the last 3 years of the contract. The years inbetween do not vary as much as either tail of the contract, but the point isthat player yearly salary =/= their salary cap hit.  

As the cap hit is truly what matters for building NHL teams, and it actually helps 'normalize' player salaries across the board, that's the more important measurement I'll be using here.

## Data Cleaning & Combining

I'll be using the player performance data that I pulled from http://corsica.hockey/skater-stats/, team data from http://corsica.hockey/team-stats/, and the cap data I scrapped in Part II. This next section I'll work on cleaning the data up, and combining these three data sets. Before that however, I'd like to highlight a important distinction between the data I'm looking at.

- I've spent a great deal of time trying to match up the player performance/cap data on player name and season.
- This in cludes really digging around in the data to determine why the two sets aren't matching.
- I was able to reduce the non-matchups from about 950 to 313.
- I'm going to drop the rest of those without cap data as most of the remaining players I simply cannot find the data for anywhere online.
- Most of these players didn't play significant time for their respective team, so it shouldn't be as big of a deal.

## Explaining different play situations

In an normal NHL game, teams usually play each other with 5 skaters on the ice (3 forwards + 2 defensemen) and a goalie. This is known as 'even strength' play. However, a team can be assessed a penalty, which is an infraction for breaking a rule. The penalized team is forced to play down a man for either 2, 4, or 5 minutes, depending on the severity of the penalty. By far the most common penalty is 2 minutes though. During the ensuing 5 on 4, known as being on the 'powerplay,' play opens up significantly, and generally the team that is up a man is able to control the puck better because they have more open ice to skate, and passing lanes to distribute the puck to teammates.  

The key takeaway is that the powerplay is a distinct type of gameplay, and is played in a different style than normal 5 on 5 hockey. Coaches mix up how they deploy players, and players play different positions than they would normally. For example, often coaches will play 4 forwards and 1 defensemen on the powerplay.

I highlight this fact because I will be making a distinction between stats collected at even strength play versus those on the powerplay because of how different of a play style they both are.

All that being said, I may stick with just one set of data over another for sake of streamlining the project. Studying subject-matter material may reveal something I don't understand about the data. If I do go that direction, it will certainly be a subject I explore further post-DSI.

## Explanation of dropping Team Stats

**Note: After much careful deliberation and evaluation, I've decided to scrap using team data altogether. I'll only be evaluating player performance vs salary (% of team salary will stay though). I'll be leaving this section here though to show my work.** To be quite honest, I think I ran into an issue of project scope creep. I originally intended to see how players help their team win, and see if that factors into value, but that would be too far off track of my original "How much is a player worth based on his production? What players are undervalued and can be acquired for less?"

**Original Idea here:** I'm adding in overall team stats to get a sense of what may be contributing to a team's performance from an individual player's perspective.

## Important Advanced Metrics

I think it's important to highlight a **key assumption** first:  

It is assumed that the team that controls the puck more, generates more score chances, and therefore scores more goals, and is more likely to win the game. This is an important underlying assumption to many NHL advanced stats, and hence why they are tracked closely. Though it is not conclusively proven that better possessions = wins, I will be accepting this assumption for now.

### Diving into Key Advanced Metrics

A key metric is Corsi. Corsi serves as a proxy measure for puck possession during five-on-five play. To calculate a Corsi number, you add shots on against the opposing team's net, missed shots for and blocked shots against. Next, you subtract shots on target against, missed shots against, and blocked shots for. Corsi is important to understand which team or player is generating, or attempting to generate the most scoring opportunities. Essentially, it is a percentage of the amount of shots a player generates towards their opponent's net, minus the shots attempts that are thrown at that player's team's net.

Fenwick is another popular stat. It is essentially the same as Corsi, but it doesn't take into account blocked shots. It has been shown to be a better predictor of possession. Despite this, it isn't used as often as Corsi.

Both Corsi and Fenwick can be applied to an individual or an entire team. And both can also be applied on an offensive only (CF or Corsi For) or defensive only (CA or Corsi Against) basis.

The numbers for Corsi and Fenwick generally fall between 40 and 60%. A team (or player) with a Corsi/Fenwick of 55% and above would be considered 'elite,' whereas lower than 45% is generally considered subpar.

PDO is another popular advanced metric thrown around alot. The acroynm doesn't stand for anything actually, and is simply 5v5 shooting percentage + 5v5 save percentage. Its importance is that it is a commonly understood as a strong indicator of luck. The average PDO for the league is around 100, so a team under that number could be said to be 'unlucky,' and a team above it could be said to be 'lucky.' This is important as stats gurus interpret this is a reflection of if a team is outperforming, or underperforming, due to luck, and is due to revert back to the mean 100.

Most of the other metrics are based off of these metrics. Some extra background first: in standard play, only 5 skaters (3 forwards & 2 defensemen - note 'skaters' excludes goaltenders) from each side are allowed on the ice at a given time. However, on a normal night, each team has 18 skaters: 12 forwards and 6 defensemen. This means that there are 4 forward line combinations, and 3 defensemen pairings. In a 60 minutes game (3 periods - 20 minutes a piece), not everyone is going to get equal ice time. As a baseline example, the best forwards in the league get around 18 - 22 minutes of ice time a night, and the best defensemen will get 25 - 28 minutes (some upwards of 30). The coaches are ultimately the ones who decide how much ice time each player gets. To normalize player's ice time, and get a better sense of whose making the most of their time, Corsi/Fenwick and other advanced stats are averaged out as if each player played a full 60 minutes (so Corsi For per 60, Corsi Against per 60, etc). The formula is as such: 5v5 ‘Y’/60 = 5v5 Base Statistic Y/5v5 Time On Ice * 60

## Distinctions in Player contracts

**Note: Player contracts can pay different rates at the NHL level versus a minor league.** By this I mean, the majority of players have one-way contracts, that is, they're paid the same regardless if they play in the NHL or AHL (American Hockey League - the highest minor league behind the NHL - the NHL's farm league essentially). However, some players, ones that may just be joining the league, or simply are 'fringe' NHL players, may have two-way contracts, that pay them radically different sums of money depending on if they play the NHL or AHL. It can be a drastic difference - say for example - having a two-way contract structure such that the player will make 75,000/year at the AHL level, but 525,000/year at the NHL level, isn't unheard of in the NHL. This article does a pretty good job of explaining it: https://www.nhl.com/lightning/news/whats-the-difference-between-a-one-way-and-a-two-way-contract/c-726016.

This is an important issue to highlight as due to injuries, trades, or simply GMs trying to give their coaching staff the best/different players to work with, it is often the case that players are moved between leagues to fill roster spots and get different looks.

The issue here is the website I scrapped this data off of contains player salaries that may reflect a pro-rated pay. By this I mean, once a season is over, for two-way contract players that spent time in both the AHL and NHL, it's common to calculate his yearly cap hit by weighting his salary to reflect time spent in the AHL (where he would be making less) against time spent in the NHL (where he would be making more) on a per-game basis.

As a result, I'm going to remove players that made less than league minimum, because that means they didn't spend the entire year with their NHL club.
