# Infrastructure as Code — squelette Terraform (Étape 5, Livrable n°7)

Ce squelette décrit l'intention d'infrastructure cible (AWS) mais n'a **jamais
été appliqué** — il ne contient pas de state, pas de credentials, et n'est pas
prêt à `apply` en l'état. Avant tout `terraform apply` :

1. Renseigner `terraform.tfvars` (non versionné) avec la région, le VPC, les
   identifiants de la base de données cible, etc.
2. Configurer un backend distant (S3 + DynamoDB pour le lock) plutôt que le
   state local par défaut.
3. Fournir des credentials AWS via un profil (`AWS_PROFILE`) ou des variables
   d'environnement — jamais en dur dans les fichiers `.tf`.
4. Valider avec `terraform plan` avant tout `apply`, et faire relire le plan
   par un humain.

`main.tf` définit uniquement le provider et les variables ; les ressources
(ECS/EC2 pour l'API et le dashboard, RDS Postgres, ECR) restent à détailler
une fois le choix d'hébergement définitif validé (cf. README section 6,
étape 5 : blue/green, KPI DevOps, alerting).
