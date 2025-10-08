waypoints

[{'_id': {'user_id': '59c2706db395b292ed622d84', 'race_id': 698, 'leg_num': 1, 'ts': 1737827340000, 'action': 'wp'},
 'pos': [{'lat': -10.15676, 'lon': -35.62734, 'idx': 25}, {'lat': -11.25134, 'lon': -36.55664, 'idx': 26}]}]



//programmation 
[{'deg': -115, 'autoTwa': True, 'isProg': True, '_id': {'user_id': '59c2706db395b292ed622d84', 'race_id': 698, 'leg_num': 1, 'ts': 1737852660000, 'action': 'heading'}}, 
 {'deg': -116, 'autoTwa': True, 'isProg': True, '_id': {'user_id': '59c2706db395b292ed622d84', 'race_id': 698, 'leg_num': 1, 'ts': 1737853260000, 'action': 'heading'}},
   {'deg': -115, 'autoTwa': True, 'isProg': True, '_id': {'user_id': '59c2706db395b292ed622d84', 'race_id': 698, 'leg_num': 1, 'ts': 1737857460000, 'action': 'heading'}}]

//
dans les 2 cas on a _id   

dans le cas wp on a
action= 'wp'

dans le cas de prog on action='heading'


si on analyse le premier terme de ba on trouve  

boatactions[0]['_id']['action']='wp'
ou
boatactions[0]['_id']['action']='heading' 






boatinfos2['ba']