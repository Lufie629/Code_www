 self.df_data=pd.concat([df_train,df_dev,df_test,df_support]

train.json test.json dev.json df_support

```
{
        "ID": "17461978",
        "profile": {
            "id": "17461978 ",
            "id_str": "17461978 ",
            "name": "SHAQ ",
            "screen_name": "SHAQ ",
            "location": "Orlando, FL ",
            "profile_location": "{'id': '55b4f9e5c516e0b6', 'url': 'https://api.twitter.com/1.1/geo/id/55b4f9e5c516e0b6.json', 'place_type': 'unknown', 'name': 'Orlando, FL', 'full_name': 'Orlando, FL', 'country_code': '', 'country': '', 'contained_within': [], 'bounding_box': None, 'attributes': {}} ",
            "description": "VERY QUOTATIOUS, I PERFORM RANDOM ACTS OF SHAQNESS ",
            "url": "https://t.co/7hsiK8cCKW ",
            "entities": "{'url': {'urls': [{'url': 'https://t.co/7hsiK8cCKW', 'expanded_url': 'http://www.ShaqFuRadio.com', 'display_url': 'ShaqFuRadio.com', 'indices': [0, 23]}]}, 'description': {'urls': []}} ",
            "protected": "False ",
            "followers_count": "15349596 ",
            "friends_count": "692 ",
            "listed_count": "45568 ",
            "created_at": "Tue Nov 18 10:27:25 +0000 2008 ",
            "favourites_count": "142 ",
            "utc_offset": "None ",
            "time_zone": "None ",
            "geo_enabled": "True ",
            "verified": "True ",
            "statuses_count": "9798 ",
            "lang": "None ",
            "contributors_enabled": "False ",
            "is_translator": "False ",
            "is_translation_enabled": "False ",
            "profile_background_color": "080203 ",
            "profile_background_image_url": "http://abs.twimg.com/images/themes/theme1/bg.png ",
            "profile_background_image_url_https": "https://abs.twimg.com/images/themes/theme1/bg.png ",
            "profile_background_tile": "False ",
            "profile_image_url": "http://pbs.twimg.com/profile_images/1673907275/image_normal.jpg ",
            "profile_image_url_https": "https://pbs.twimg.com/profile_images/1673907275/image_normal.jpg ",
            "profile_link_color": "2FC2EF ",
            "profile_sidebar_border_color": "181A1E ",
            "profile_sidebar_fill_color": "252429 ",
            "profile_text_color": "666666 ",
            "profile_use_background_image": "True ",
            "has_extended_profile": "False ",
            "default_profile": "False ",
            "default_profile_image": "False "
        },
        "tweet": [
            n条完整tweet文本
        ],
        "neighbor": null,
        "domain": [
            "Politics",
            "Business",
            "Entertainment"
        ],
        "label": "0"
    },
```

df_train: ID profile tweet neighbor domain label

```
 ID                                            profile  ...                                           neighbor label
0                17461978  {'id': '17461978 ', 'id_str': '17461978 ', 'na...  ...                                               None    
 0
1     1297437077403885568  {'id': '1297437077403885568 ', 'id_str': '1297...  ...  {'following': ['170861207', '23970102', '47293...    
 1
2                17685258  {'id': '17685258 ', 'id_str': '17685258 ', 'na...  ...  {'following': ['46464108', '21536398', '186434...    
 0
3                15750898  {'id': '15750898 ', 'id_str': '15750898 ', 'na...  ...  {'following': ['2324715174', '24030137', '2336...    
 0
4              1659167666  {'id': '1659167666 ', 'id_str': '1659167666 ',...  ...  {'following': ['1628313708', '726405625', '130...    
 1
...                   ...                                                ...  ...                                                ...   ...
8273           1630890068  {'id': '1630890068 ', 'id_str': '1630890068 ',...  ...  {'following': ['237453978', '462581299', '1706...    
 0
8274   713519580757536768  {'id': '713519580757536769 ', 'id_str': '71351...  ...  {'following': ['36991422', '32567081', '133983...    
 1
8275             93345260  {'id': '93345260 ', 'id_str': '93345260 ', 'na...  ...  {'following': ['714636670268792832', '23341114...    
 1
8276           1749309397  {'id': '1749309397 ', 'id_str': '1749309397 ',...  ...  {'following': ['3124065581', '413364940', '211...     1
8277             50471224  {'id': '50471224 ', 'id_str': '50471224 ', 'na...  ...  {'following': ['4202878276', '637216245', '129...     1
```



label.pt：存储了每个节点的身份信息

description.pt ：利用bert将描述变成768维向量

tweets_tensor.pt:同上

followers_count.pt：每个用户的关注人个数 followers_count=torch.tensor(np.array(followers_count,dtype=np.float32)).to(self.device)

followers_count.pt、friends_count.pt、favourites_count.pt、active_days.pt、statuses_count.pt同上，分别标准化之后cat，存成num_properties_tensor.pt

cat_properties_tensor.pt：profile中的余下部分信息

edge_index.pt：所有user节点

edge_type.pt