terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Backend distant à configurer avant tout usage réel (cf. README.md de ce dossier) :
  # backend "s3" {
  #   bucket         = "dst-airlines-terraform-state"
  #   key            = "dst-airlines/terraform.tfstate"
  #   region         = var.aws_region
  #   dynamodb_table = "dst-airlines-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "Région AWS de déploiement"
  type        = string
  default     = "eu-west-3"
}

variable "environment" {
  description = "Environnement (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Préfixe de nommage des ressources"
  type        = string
  default     = "dst-airlines"
}

# Ressources (ECR, ECS/EC2, RDS, ...) à ajouter une fois l'architecture
# de déploiement validée — non définies ici pour éviter tout `apply`
# accidentel contre un compte AWS réel.
