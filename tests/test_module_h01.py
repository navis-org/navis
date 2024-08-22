import navis.interfaces.h01 as h01

# def test_auth():
#     auth = h01.Authentication()
#     auth.setup_token(make_new=False)

# def test_get_exist_token():
#     client = h01.get_cave_client()
#     print(client.auth.token)
#     table = client.materialize.get_tables()
#     print(table)
    # try:
    #     table = client.materialize.get_tables()
    #     print(table)
    # except Exception as e:
    #     print("Save your token using auth.save_token(token="", overwrite=True)")
    #     # auth.save_token(token="", overwrite=True)

def test_get_cave_client():
    client = None
    # client = h01.get_cave_client_with_token(token="3ac8d9f0de1b192dba9c085114f0f811")
    client = h01.get_cave_client()
        
    tables = client.materialize.get_tables()
#     tables =
    print(tables)
    assert len(tables) == 5, "current # of tables is 5"

def test_get_cloudvol():
    url = 'graphene://https://local.brain-wire-test.org/segmentation/1.0/h01_full0_v2',
    cv = h01.get_cloudvol(url)

    root_id = '864691132320931868
    cv.mesh.save(root_id)
