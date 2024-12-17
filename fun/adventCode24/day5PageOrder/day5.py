

with open('input.txt', 'r') as text:
    inputList = text.read().splitlines()

    rules = inputList[:inputList.index('')]
    rulesList = [rule.split('|') for rule in rules]
    #print(rulesList)
    pageOrder = inputList[inputList.index('') + 1:]


    def is_good(pages):
        for x , y in rulesList:
            if x in pages and y in pages:
                if pages.index(x) > pages.index(y):
                    return False
            
        return True

    goodPageOrders = []
    for pages in pageOrder:
        pageOrderList = pages.split(',')
        if is_good(pageOrderList):
            goodPageOrders.append(pageOrderList)
        

    #part1
    middleSum = sum(int(pages[len(pages) // 2]) for pages in goodPageOrders)

    #part2
    pageOrderListFull = [page.split(',') for page in pageOrder]
    incorrectPageOrder = [x for x in pageOrderListFull if x not in goodPageOrders]
    
    def make_good(pages):
        changes_made = True
        while changes_made:
            changes_made = False
            for x, y in rulesList:
                if x in pages and y in pages:
                    if pages.index(x) > pages.index(y):
                        pages[pages.index(x)], pages[pages.index(y)] = pages[pages.index(y)], pages[pages.index(x)]
                        changes_made = True
        return pages


    incorrectCorrected = [make_good(pages) for pages in incorrectPageOrder]
    middleSum2 = sum(int(pages[len(pages) // 2]) for pages in incorrectCorrected)

    print(middleSum2)

    # rulesDic = {}
    # for rule in rules:
    #     rulesList = rule.split('|')

    #     if rulesList[0] in rulesDic:
    #         rulesDic[rulesList[0]].append(rulesList[1])
    #     else:
    #         rulesDic[rulesList[0]] = [rulesList[1]]
    # goodPageOrders = []
    # for pages in pageOrder:
    #     pageOrderList = pages.split(',')
    #     for idx, page in enumerate(pageOrderList):
    #         if page in rulesDic:
    #             if all(x in pageOrderList[idx:] for x in rulesDic[page]) and not(any(y in pageOrderList[:idx] for y in rulesDic[page])):
    #                 goodPageOrders.append(pages)