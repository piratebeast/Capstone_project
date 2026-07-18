using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SkincareAdvisor.Domain.Entities;
using SkincareAdvisor.Application.Interfaces;

namespace SkincareAdvisor.Application.Interfaces
{
    public interface IScanCritiqueService
    {
        Task<ScanCritique> GenerateCritiqueAsync(ScanHistory scan);
    }
}
