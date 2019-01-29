User stats
----------
Pymaid has functions that let you look at stats/contributions of individual
users as well as your project's history and logs.

There are quite a few simple examples in the documentation of each function.
Examples given here are a bit more elaborate.

In this first example, we will compare two users:

>>> # This assumes you have imported pymaid and setup a CATMAID instance
>>> import matplotlib.pyplot as plt
>>> # Get history for 3 months
>>> hist = pymaid.get_history(start_date=(2017,1,1), end_date=(2017,3,31))
>>> # Create empty plot
>>> fig, ax = plt.subplots( 3, 1, sharex=True )
>>> # Plot cable length in top plot
>>> hist.cable.loc[['user_id1', 'user_id2']].T.plot(ax=ax[0])
>>> ax[0].set_ylabel('cable traced [nm]')
>>> # Plot connector links created in middle plot
>>> hist.connector_links.loc[['user_id1', 'user_id2']].T.plot(ax=ax[1], legend=False)
>>> ax[1].set_ylabel('links created')
>>> # Plot nodes reviewed in bottom plot
>>> hist.cable.loc[['user_id1', 'user_id2']].T.plot(ax=ax[2], legend=False)
>>> ax[2].set_ylabel('nodes reviewed')
>>> # Tighten plot
>>> plt.tight_layout()
>>> # Render plot
>>> plt.show()

Next, we will check contributions that users have made to a set of neurons.
This comes in handy e.g. when deciding who to include in the author list.

>>> nl = pymaid.find_neurons(annotations='glomerulus DA1')
>>> # Get contributions in number of nodes/links/reviews
>>> cont = pymaid.get_user_contributions( nl )
>>> # Note that I've scrambled the user column 
>>> cont.head()
       user  nodes  presynapses  postsynapses
0  aaaaaaaa  47880         3854          1911
1    bbbbbb   5930          403           133
2  cccccccc   5204           87             4
3    dddddd   4803           89           178
4   eeeeeee   4267           61            11

Note that one user appears to have placed the majority of nodes and connectors
for this set of neurons. You can get the percentage via:

>>> cont.loc[:, ['nodes', 'presynapses', 'postsynapses']] / cont[['nodes', 'presynapses', 'postsynapses']].sum(axis=0)
       nodes  presynapses  postsynapses
0   0.456666     0.602324      0.720000
1   0.057295     0.024322      0.120000
2   0.056335     0.062122      0.049720
3   0.049455     0.013478      0.001495

Sometimes mere number of nodes and connectors can be misleading. For example,
when people have been tracing in backbone vs fine dendrites. In those cases,
looking at the time invested makes more sense:

>>> # Get time invested [minutes]
>>> inv = pymaid.get_time_invested( nl )
>>> inv.head()
              total  creation  edition  review
user                                          
aaaaaaaa       3033      2250     1620     243
eeeeeee         627       183      180     333
ffffffffffff    381       108       15     144
bbbbbb          357       219      189       0
cccccccc        333       147       72     126
>>> # Create bar plot for top 10 contributors
>>> import matplotlib.pyplot as plt
>>> ax = inv.ix[:10].plot.bar()
>>> plt.show()

Please note that all time-related metrics are always in "CATMAID time" which
is generally about 2-3x less than real-time! 


Reference
=========

.. autosummary::
    :toctree: generated/

	~pymaid.get_user_contributions 
	~pymaid.get_time_invested
	~pymaid.get_history
	~pymaid.get_logs
	~pymaid.get_contributor_statistics
	~pymaid.get_user_list
	~pymaid.get_user_actions
	~pymaid.get_transactions
    ~pymaid.get_team_contributions
    