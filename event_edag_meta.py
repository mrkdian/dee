EVENT_TYPES = ['EquityFreeze', 'EquityRepurchase', 'EquityUnderweight', 'EquityOverweight', 'EquityPledge']
EVENT_FIELDS = {
    'EquityFreeze': (
        ['EquityHolder', 'FrozeShares', 'LegalInstitution', 'TotalHoldingShares',
        'TotalHoldingRatio', 'StartDate', 'EndDate', 'UnfrozeDate'],
        ['EquityHolder', 'FrozeShares', 'LegalInstitution', 'TotalHoldingShares',
        'TotalHoldingRatio', 'StartDate', 'EndDate', 'UnfrozeDate']
    ),
    'EquityRepurchase': (
        ['CompanyName', 'HighestTradingPrice', 'LowestTradingPrice', 'RepurchasedShares',
        'ClosingDate', 'RepurchaseAmount',],
        ['CompanyName', 'HighestTradingPrice', 'LowestTradingPrice', 'RepurchasedShares',
        'ClosingDate', 'RepurchaseAmount',]
    ),
    'EquityUnderweight': (
        ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
        ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice']
    ),
    'EquityOverweight': (
        ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
        ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice']
    ),
    'EquityPledge': (
        ['Pledger', 'PledgedShares', 'Pledgee', 'TotalHoldingShares', 'TotalHoldingRatio', 'TotalPledgedShares',
        'StartDate', 'EndDate', 'ReleasedDate'],
        ['Pledger', 'PledgedShares', 'Pledgee', 'TotalHoldingShares', 'TotalHoldingRatio', 'TotalPledgedShares',
        'StartDate', 'EndDate', 'ReleasedDate']
    )
}
EVENT_TYPE2ID = {}

NER_LABEL_LIST = ['O']
NER_LABEL2ID = {'O': 0}
for i, ee_type in enumerate(EVENT_TYPES):
    EVENT_TYPE2ID[ee_type] = i

    ee_roles, ee_role_types = EVENT_FIELDS[ee_type]
    for ee_role, ee_role_type in zip(ee_roles, ee_role_types):
        if 'B-' + ee_role in NER_LABEL_LIST:
            continue
        NER_LABEL_LIST.append('B-' + ee_role)
        NER_LABEL2ID[NER_LABEL_LIST[-1]] = len(NER_LABEL_LIST) - 1
        NER_LABEL_LIST.append('I-' + ee_role)
        NER_LABEL2ID[NER_LABEL_LIST[-1]] = len(NER_LABEL_LIST) - 1

EVENT_TYPE_FIELDS_PAIRS = []
for event_type in EVENT_TYPES:
    fields = EVENT_FIELDS[event_type][0]
    EVENT_TYPE_FIELDS_PAIRS.append((event_type, fields))