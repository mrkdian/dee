EVENT_TYPES = ['破产清算', '重大安全事故', '股东减持', '股权质押', '股东增持', '股权冻结', '高层死亡', '重大资产损失', '重大对外赔付']
EVENT_FIELDS = {
    '破产清算': (['公司名称', '公告时间', '受理法院', '裁定时间', '公司行业'], ['公司', '时间', '机构', '时间', '行业']),
    '重大安全事故': (['公司名称', '公告时间', '伤亡人数', '损失金额', '其他影响'], ['公司', '时间', '数字', '数字', '文本短语']),
    '股东减持': (['减持的股东', '减持金额', '减持开始日期'], ['公司/人名', '数字和单位', '时间']),
    '股权质押': (['质押金额', '质押开始日期', '接收方', '质押方', '质押结束日期'], ['数字', '时间', '公司/人名', '公司/人名', '时间']),
    '股东增持': (['增持的股东', '增持金额', '增持开始日期'], ['公司/人名', '数字和单位', '时间']),
    '股权冻结': (['被冻结股东', '冻结金额', '冻结开始日期', '冻结结束日期'], ['公司/人名', '数字', '时间', '时间']),
    '高层死亡': (['公司名称', '高层人员', '高层职务', '死亡/失联时间', '死亡年龄'], ['公司', '人名', '职称', '时间', '数字']),
    '重大资产损失': (['公司名称', '公告时间', '损失金额', '其他损失'], ['公司', '时间', '数字', '文本短语']),
    '重大对外赔付': (['公告时间', '公司名称', '赔付对象', '赔付金额'], ['时间', '公司', '公司/人名', '数字'])
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