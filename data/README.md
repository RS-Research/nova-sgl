# Dataset Folder

Datasets are included in this repository.

For the LastFM social experiment place them as follows:

```text
data/
└── lastfm/
    ├── lastfm.inter
    ├── lastfm.item
    └── lastfm.net
	
```

# Expected LastFM headers:

```text
lastfm.inter:
user_id:token    artist_id:token

lastfm.item:
artist_id:token  name:token_seq  url:token  pictureURL:token

lastfm.net:
source_id:token  target_id:token
```
# Note:
The lastfm.net file is required for the full social version of NOVA-SGL.