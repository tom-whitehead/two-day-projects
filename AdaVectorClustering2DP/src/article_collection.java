public static void main(String[] args) throws Exception {
    Base.setEnvironment(Environment.PRODUCTION);
    Base.setDbConnectionLevel(DBConnectionLevel.READONLY);
    Base.initialiseDBConnections(new SimpleShutdownMonitor());
  
    Language language;
    if (args.length == 1) {
      language = Language.getFromISO639Dot1(args[0]);
    } else {
      language = Language.ENGLISH;
    }

    long end = UnixTime.now() - 3600 * 2;
    long start = end - 86_400;
    System.out.println(start);
    System.out.println(end);

    try (Connection con = Base.getSQLDb().getConnection()) {
      List<Account> properties = AccountStatic.selectAllByPropertyState(con, PropertyState.ACTIVE);
      properties = properties.stream().filter(p -> !p.hasTimeEnd())
          .filter(p -> language.equals(p.getLanguage())).collect(Collectors.toList());
      System.out.println("Number of properties: " + properties.size());


      Set<Article> articles = new HashSet<>();
      int count = 0;

      for (Account property : properties) {

        AccountSources accountSources =
            AccountService.getInstance().getAccountSources(property.getAccountId());
        Set<SyndFeedId> syndFeedIds = accountSources.getSyndFeedIds();

        Set<ArticleId> articleIds =
            MediaItemSourceService.getArticleIdsByPublishTime(start, end, syndFeedIds,
                AccountStatic.selectAllAccountAPIIds(con, property.getAccountId(),
                    APIType.SOCIAL_APIS), Integer.MAX_VALUE);

        List<Article> propertyArticles =
            ArticleService.getArticleByIds(articleIds, NoSQLFields.ALL);
        articles.addAll(propertyArticles);

        count++;
        if (count % 50 == 0) {
          System.out.println("Properties processed: " + count
              + "; articles collected: " + articles.size());
        }
      }

      System.out.println("Total articles collected: " + articles.size());

      Map<ArticleId, Long> publishTimes = MediaItemSourceService.getArticlesInitialPublishTime(
              articles.stream().map(Article::getArticleId).collect(Collectors.toSet()),
              Sources.ALL_ARTICLE_SOURCE_TYPES);

      List<List<String>> data = new ArrayList<>();
      for (Article article : articles) {

        if (article.getArticleContent() == null) {
          continue;
        }

        ArticleContent articleContent = article.getArticleContent();
        if (StringUtils.isBlank(articleContent.getText())) {
          continue;
        }

        List<String> row = new ArrayList<>();
        row.add(article.getArticleId().toString());
        row.add(articleContent.getTitle());
        row.add(articleContent.getText());
        row.add(articleContent.getLanguage());
        row.add(String.valueOf(publishTimes.getOrDefault(article.getArticleId(), 0L)));
        long timeCreated = article.getTimeCreated();
        row.add(String.valueOf(timeCreated));
        data.add(row);
      }

      String fileName =
          language.getLowerCaseTwoLetterCode() + "_articles_" + start + "_" + end + ".csv";
      writeToCsv(data, fileName);

    }
}